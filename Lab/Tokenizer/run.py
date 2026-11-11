"""Download, train/benchmark, and inspect a tokenizer from the command line."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import sys
import unicodedata
from time import perf_counter
from urllib.error import URLError
from urllib.request import urlopen

from bpe import NaiveBPE, OptimizedBPE

ROOT = Path(__file__).resolve().parent
CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
MODELS = {"naive": NaiveBPE, "optimized": OptimizedBPE}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    def width(text: str) -> int:
        return sum(0 if unicodedata.combining(char) else
                   2 if unicodedata.east_asian_width(char) in "WF" else 1
                   for char in text)

    widths = [max(width(row[i]) for row in [headers, *rows])
              for i in range(len(headers))]
    border = "+" + "+".join("-" * (size + 2) for size in widths) + "+"

    def line(row: list[str]) -> str:
        return "| " + " | ".join(cell + " " * (size - width(cell))
                                 for cell, size in zip(row, widths)) + " |"

    print(border)
    print(line(headers))
    print(border)
    for row in rows:
        print(line(row))
    print(border)


def show_encoding(model, text: str, details: bool = False) -> None:
    result = model.inspect(text)
    if details:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        visible = {" ": "␠", "\n": r"\n", "\r": r"\r", "\t": r"\t"}
        rows = []
        for position, item in enumerate(result["tokens"], 1):
            label = "".join(visible.get(char, char if char.isprintable()
                                        else f"\\u{ord(char):04x}")
                            for char in item["token"])
            rows.append([str(position), label, str(item["id"])])
        print_table(["#", "Token", "ID"], rows)


def download(args: argparse.Namespace) -> None:
    existing = args.output.exists()
    if existing:
        content = args.output.read_bytes()
    else:
        with urlopen(CORPUS_URL, timeout=30) as response:
            content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != CORPUS_SHA256:
        raise ValueError("Corpus checksum mismatch; no file was overwritten. "
                         "Use --corpus with your own UTF-8 text instead.")
    text = content.decode("utf-8")
    if not text.startswith("First Citizen:"):
        raise ValueError("Unexpected corpus content; download was not saved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not existing:
        args.output.write_bytes(content)
    metadata = {"name": "Tiny Shakespeare", "url": CORPUS_URL,
                "sha256": digest,
                "bytes": len(content), "characters": len(text)}
    write_json(args.output.with_suffix(".source.json"), metadata)
    if args.details:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"{'Already downloaded and verified' if existing else 'Saved'}: {args.output}")


def benchmark(args: argparse.Namespace) -> None:
    if args.merges < 0 or args.min_frequency < 1 or args.repeats < 1 or args.max_chars < 0:
        raise ValueError("Invalid merges/min-frequency/repeats/max-chars")
    # Preserve CRLF if present: encoding/decoding is lossless, no normalization.
    with args.corpus.open(encoding="utf-8", newline="") as source:
        corpus = source.read()
    if not corpus:
        raise ValueError("Corpus is empty")
    if args.max_chars:
        corpus = corpus[:args.max_chars]
    # Split on a line boundary when possible; held-out text never enters training.
    split = int(len(corpus) * 0.9)
    boundary = corpus.rfind("\n", 0, split + 1)
    if boundary > 0:
        split = boundary + 1
    train_text, held_out = corpus[:split], corpus[split:]
    if not train_text or not held_out:
        raise ValueError("Need enough corpus text to create nonempty train/test splits")

    report = {"python": platform.python_version(), "platform": platform.platform(),
              "corpus": str(args.corpus.resolve()),
              "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
              "train_characters": len(train_text), "test_characters": len(held_out),
              "requested_merges": args.merges, "min_frequency": args.min_frequency,
              "repeats": args.repeats, "implementations": {}}
    trained = {}
    for name, cls in MODELS.items():
        train_times, encode_times = [], []
        print(f"Training {name}...", flush=True)
        for _ in range(args.repeats):
            start = perf_counter()
            model = cls().train(train_text, args.merges, args.min_frequency)
            train_times.append(perf_counter() - start)
            start = perf_counter()
            ids = model.encode(held_out)
            encode_times.append(perf_counter() - start)
            if model.decode(ids) != held_out:
                raise AssertionError(f"{name}: failed held-out roundtrip")
        trained[name] = (model, ids)
        model.save(args.output / f"{name}.json")
        report["implementations"][name] = {
            "train_seconds": train_times, "train_median_seconds": median(train_times),
            "encode_seconds": encode_times, "encode_median_seconds": median(encode_times),
            "vocabulary_size": len(model.vocab), "merges": len(model.merges),
            "held_out_tokens": len(ids), "roundtrip_ok": True,
            "work_counters": model.stats,
        }
    naive, fast = trained["naive"][0], trained["optimized"][0]
    if (naive.vocab != fast.vocab or naive.merges != fast.merges
            or trained["naive"][1] != trained["optimized"][1]):
        raise AssertionError("Implementations disagree on vocabulary, merges, or held-out IDs")
    report["identical_vocabulary_merges_and_ids"] = True
    baseline = report["implementations"]["naive"]
    optimized = report["implementations"]["optimized"]
    report["training_speedup"] = baseline["train_median_seconds"] / optimized["train_median_seconds"]
    report["encoding_speedup"] = baseline["encode_median_seconds"] / optimized["encode_median_seconds"]
    write_json(args.output / "benchmark.json", report)
    if args.details:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_table(["Version", "Train (s)", "Encode (ms)"], [
            [name, f"{row['train_median_seconds']:.4f}",
             f"{row['encode_median_seconds'] * 1000:.3f}"]
            for name, row in report["implementations"].items()
        ])
        print(f"Training speedup: {report['training_speedup']:.2f}x; "
              f"encoding speedup: {report['encoding_speedup']:.2f}x")
        print("Vocabulary, merges and IDs match; decoding restores the original text.")
    print(f"Models and report: {args.output}")
    if args.text is not None:
        show_encoding(fast, args.text, args.details)


def encode(args: argparse.Namespace) -> None:
    model = MODELS[args.implementation].load(args.model)
    if args.text is not None:
        text = args.text
    else:
        with args.text_file.open(encoding="utf-8", newline="") as source:
            text = source.read()
    show_encoding(model, text, args.details)


def main() -> None:
    # Windows consoles can otherwise fail when displaying Chinese/emoji.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="Download the public Tiny Shakespeare text")
    get.add_argument("--output", type=Path, default=ROOT / "data/tinyshakespeare.txt")
    get.add_argument("--details", action="store_true", help="Show full JSON details")
    get.set_defaults(func=download)
    bench = commands.add_parser("benchmark", help="Train both versions and verify equivalence")
    bench.add_argument("--corpus", type=Path, default=ROOT / "data/tinyshakespeare.txt")
    bench.add_argument("--merges", type=int, default=256)
    bench.add_argument("--min-frequency", type=int, default=2)
    bench.add_argument("--max-chars", type=int, default=100_000,
                       help="Limit corpus characters for a quick run; 0 uses the complete file")
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument("--output", type=Path, default=ROOT / "artifacts")
    bench.add_argument("--text", help="Also inspect this text after training")
    bench.add_argument("--details", action="store_true", help="Show full JSON details")
    bench.set_defaults(func=benchmark)
    enc = commands.add_parser("encode", help="Load a saved model and show tokens and IDs; no training")
    enc.add_argument("--model", type=Path, default=ROOT / "artifacts/optimized.json")
    enc.add_argument("--implementation", choices=MODELS, default="optimized")
    enc.add_argument("--details", action="store_true", help="Also show bytes and decoded text as JSON")
    source = enc.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path, help="Read the complete UTF-8 text file")
    enc.set_defaults(func=encode)
    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, URLError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

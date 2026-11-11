"""Two readable, equivalent byte-level BPE implementations (standard library only)."""

from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import json
from pathlib import Path
import re

Pair = tuple[int, int]
Merge = tuple[int, int, int]
# Keep all whitespace, including newlines. Never merge across these boundaries.
PIECES = re.compile(r"\s+|\S+")
FORMAT = "byte-bpe-experiment-v1"


def split_text(text: str) -> list[bytes]:
    return [match.group().encode("utf-8") for match in PIECES.finditer(text)]


def pair_counts(ids: list[int]) -> Counter[Pair]:
    return Counter(zip(ids, ids[1:]))


def merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Merge non-overlapping occurrences from left to right, including a a a."""
    result = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


class BPETokenizer:
    """Shared vocabulary, serialization, and lossless UTF-8 decoding."""

    def __init__(self) -> None:
        self.vocab = [bytes([i]) for i in range(256)]
        self.merges: list[Merge] = []
        self.ranks: dict[Pair, tuple[int, int]] = {}
        self.stats: dict[str, int] = {}

    def _reset(self, num_merges: int, min_frequency: int) -> None:
        if num_merges < 0 or min_frequency < 1:
            raise ValueError("num_merges must be >= 0; min_frequency must be >= 1")
        self.__init__()
        self.stats = {"pair_visits": 0, "pieces_visited": 0}

    def _add_merge(self, pair: Pair) -> int:
        new_id = len(self.vocab)
        self.ranks[pair] = (len(self.merges), new_id)
        self.merges.append((*pair, new_id))
        self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])
        return new_id

    def train(self, text: str, num_merges: int = 256,
              min_frequency: int = 2) -> BPETokenizer:
        raise NotImplementedError

    def _encode_piece(self, ids: list[int]) -> list[int]:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return [token for piece in split_text(text)
                for token in self._encode_piece(list(piece))]

    def decode(self, ids: list[int]) -> str:
        if any(type(i) is not int or i < 0 or i >= len(self.vocab) for i in ids):
            raise ValueError("Token ID outside vocabulary")
        # Decode once after joining: a single token may contain only part of a character.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8")

    def inspect(self, text: str) -> dict:
        ids = self.encode(text)
        tokens = []
        for token_id in ids:
            raw = self.vocab[token_id]
            try:
                label = raw.decode("utf-8")
            except UnicodeDecodeError:
                # Do not silently replace partial UTF-8 characters with the replacement character.
                label = "".join(f"\\x{value:02x}" for value in raw)
            tokens.append({"id": token_id, "token": label, "bytes_hex": raw.hex()})
        restored = self.decode(ids)
        return {"text": text, "tokens": tokens, "ids": ids,
                "decoded": restored, "roundtrip_ok": restored == text}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"format": FORMAT, "merges": self.merges},
                                     indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError("Unsupported tokenizer file format")
        model = cls()
        for item in data["merges"]:
            if (not isinstance(item, list) or len(item) != 3
                    or any(type(value) is not int for value in item)):
                raise ValueError("Invalid merge record")
            a, b, new_id = item
            if (not 0 <= a < len(model.vocab) or not 0 <= b < len(model.vocab)
                    or new_id != len(model.vocab) or (a, b) in model.ranks):
                raise ValueError("Invalid merge IDs")
            model._add_merge((a, b))
        return model


class NaiveBPE(BPETokenizer):
    """Baseline: scan every original piece and recount every pair on every round."""

    def train(self, text: str, num_merges: int = 256,
              min_frequency: int = 2) -> NaiveBPE:
        self._reset(num_merges, min_frequency)
        words = [list(piece) for piece in split_text(text)]
        for _ in range(num_merges):
            counts: Counter[Pair] = Counter()
            for word in words:
                counts.update(zip(word, word[1:]))
                self.stats["pair_visits"] += max(0, len(word) - 1)
            self.stats["pieces_visited"] += len(words)
            if not counts:
                break
            # Explicit tie rule: smallest (left ID, right ID), identical in both trainers.
            best = min(counts, key=lambda pair: (-counts[pair], pair))
            if counts[best] < min_frequency:
                break
            new_id = self._add_merge(best)
            words = [merge_pair(word, best, new_id) for word in words]
        return self

    def _encode_piece(self, ids: list[int]) -> list[int]:
        while len(ids) > 1:
            candidates = [(self.ranks[pair][0], pair)
                          for pair in zip(ids, ids[1:]) if pair in self.ranks]
            if not candidates:
                break
            _, best = min(candidates)
            ids = merge_pair(ids, best, self.ranks[best][1])
        return ids


class OptimizedBPE(BPETokenizer):
    """Weighted deduplication + inverted index + incremental counts + lazy heap."""

    def train(self, text: str, num_merges: int = 256,
              min_frequency: int = 2) -> OptimizedBPE:
        self._reset(num_merges, min_frequency)
        word_counts = Counter(split_text(text))
        words = [list(word) for word in word_counts]
        weights = list(word_counts.values())
        counts: Counter[Pair] = Counter()
        index: dict[Pair, set[int]] = defaultdict(set)
        versions: dict[Pair, int] = defaultdict(int)

        for word_id, word in enumerate(words):
            for pair, occurrences in pair_counts(word).items():
                counts[pair] += occurrences * weights[word_id]
                index[pair].add(word_id)
            self.stats["pair_visits"] += max(0, len(word) - 1)
        self.stats["pieces_visited"] += len(words)
        heap = [(-count, pair, 0) for pair, count in counts.items()]
        heapq.heapify(heap)

        for _ in range(num_merges):
            while heap:
                negative_count, best, version = heapq.heappop(heap)
                if version == versions[best] and -negative_count == counts[best]:
                    break
            else:
                break
            if counts[best] < min_frequency:
                break
            new_id = self._add_merge(best)
            changed: set[Pair] = set()

            # Copy because index[best] is updated while these words are processed.
            for word_id in list(index[best]):
                before = words[word_id]
                after = merge_pair(before, best, new_id)
                old_pairs, new_pairs = pair_counts(before), pair_counts(after)
                words[word_id] = after
                self.stats["pieces_visited"] += 1
                self.stats["pair_visits"] += max(0, len(before) - 1) + max(0, len(after) - 1)
                for pair in old_pairs.keys() | new_pairs.keys():
                    delta = new_pairs[pair] - old_pairs[pair]
                    if delta:
                        counts[pair] += delta * weights[word_id]
                        changed.add(pair)
                    if pair in old_pairs and pair not in new_pairs:
                        index[pair].discard(word_id)
                    elif pair in new_pairs and pair not in old_pairs:
                        index[pair].add(word_id)

            for pair in changed:
                versions[pair] += 1
                if counts[pair] > 0:
                    heapq.heappush(heap, (-counts[pair], pair, versions[pair]))
        return self

    def _encode_piece(self, ids: list[int]) -> list[int]:
        """Min-heap by merge rank, array-backed linked neighbors, stale-entry checks."""
        size = len(ids)
        if size < 2:
            return ids
        previous = [i - 1 for i in range(size)]
        following = [i + 1 for i in range(size)]
        following[-1] = -1
        alive = [True] * size
        heap = []

        def push(left: int) -> None:
            if left < 0 or not alive[left]:
                return
            right = following[left]
            if right == -1:
                return
            rule = self.ranks.get((ids[left], ids[right]))
            if rule is not None:
                rank, new_id = rule
                heapq.heappush(heap, (rank, left, right, ids[left], ids[right], new_id))

        for i in range(size - 1):
            push(i)
        while heap:
            _, left, right, old_left, old_right, new_id = heapq.heappop(heap)
            if (not alive[left] or not alive[right] or following[left] != right
                    or ids[left] != old_left or ids[right] != old_right):
                continue
            ids[left] = new_id
            alive[right] = False
            neighbor = following[right]
            following[left] = neighbor
            if neighbor != -1:
                previous[neighbor] = left
            push(previous[left])
            push(left)
        result = []
        position = 0
        while position != -1:
            result.append(ids[position])
            position = following[position]
        return result

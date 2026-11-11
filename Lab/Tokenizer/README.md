# BPE Tokenizer 对比实验

使用 Python 3.10+ 标准库，对比基础版的全量扫描与优化版的加权去重、索引、增量统计和堆。

## 运行

在 `Lab/Tokenizer` 目录下执行：

```powershell
# 首次使用：下载语料并训练两版
python run.py download
python run.py benchmark

# 读取指定的 UTF-8 文本文件
python run.py encode --text-file "input.txt"

# 也可直接传入文本
python run.py encode --text "lower lowest"
```

将 `input.txt` 放在本目录，或指定其他相对路径、绝对路径。读取整个文件并保留空格和换行，加载已有模型，不重新训练。

输入内容为 `lower lowest` 时，输出示例：

```text
+---+-------+-----+
| # | Token | ID  |
+---+-------+-----+
| 1 | l     | 108 |
| 2 | ow    | 295 |
| 3 | er    | 268 |
| 4 | ␠     | 32  |
| 5 | l     | 108 |
| 6 | ow    | 295 |
| 7 | est   | 403 |
+---+-------+-----+
```

`␠` 表示空格，`\n`、`\t` 表示换行和制表符；加 `--details` 查看完整 JSON。

## 训练设置

默认使用 [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) 英文语料的前 10 万字符，约 90% 训练、10% 测试，进行 256 次合并。

可用 `--corpus 文件路径` 指定自己的 UTF-8 语料，`--max-chars 0` 使用全文，`--merges 512` 调整合并次数，`--repeats 3` 重复计时。

模型和报告保存在 `artifacts/`。程序会检查两版词表、合并规则、测试集 ID 一致，并验证解码能还原原文。只有重新运行 `benchmark` 才会重新训练。

## 文件

- `bpe.py`：基础版与优化版算法。
- `run.py`：下载、训练对比和文本编码入口。

使用字节级 BPE，ID 与笔记中的示意编号不同；英文语料未学到的中文片段可能显示为字节转义。语料、模型和缓存已通过 `.gitignore` 排除。

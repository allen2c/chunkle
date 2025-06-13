# chunkle

Split big text into reader‑friendly pieces while respecting **line** and **token** budgets.

## Install

```bash
pip install chunkle
```

Compatible with Python ≥ 3.11.

## Quick start

```python
from chunkle import chunk

for part in chunk(big_text, lines_per_chunk=20, tokens_per_chunk=500):
    ...  # stream, save, or send
```

The generator yields a chunk the moment **both** budgets are met.

### **Defaults**

* `lines_per_chunk = 20`
* `tokens_per_chunk = 500`

## API

```python
def chunk(
    content: str,
    *,
    lines_per_chunk: int = 20,
    tokens_per_chunk: int = 500,
    encoding: tiktoken.Encoding | None = None,
) -> typing.Generator[str, None, None]:
    ...
```

## Comming Next

* **Benchmark** batched vs. per‑char tokenization on a 10 MB multilingual file.
* Ship **0.1.1** with CRLF handling and an expanded README.
* Add a **GitHub Action** matrix (Python 3.11 & 3.12) to prevent regressions.

## License

MIT © 2025 Allen Chou

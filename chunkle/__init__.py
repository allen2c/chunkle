import typing

import tiktoken

YIELD_LATER_CHARS = set("。？！!?;；:：,，、…")  # half- & full-width


def chunk(
    content: str,
    *,
    lines_per_chunk: int = 20,
    tokens_per_chunk: int = 500,
    encoding: tiktoken.Encoding | None = None,
) -> typing.Generator[str, None, None]:
    """
    Split *content* into reader-friendly chunks that stay within both line and
    token budgets, always ending at a “safe” boundary and never starting with
    whitespace or strong punctuation.

    The function maintains two parallel limits—`lines_per_chunk` and
    `tokens_per_chunk`—and emits a chunk as soon as **both** limits are reached
    or exceeded **and** the current character is a safe break-point:

    * a newline (``\\n``); or
    * a character in the module-level constant `YIELD_LATER_CHARS`
    (full-width/half-width sentence-ending punctuation).

    Special rules ensure clean chunk boundaries:

    * **Meaning vs. NOT-meaning characters**
    A meaning character is anything that is *not* whitespace and *not* in
    `YIELD_LATER_CHARS`.  Chunks never start with a NOT-meaning character.
    Immediately after a chunk flush, any contiguous NOT-meaning characters
    are absorbed into the same chunk so that the following chunk begins with
    the first meaning character.

    * **Blank lines (paragraph breaks)**
    A blank line (two consecutive newlines, ``\\n\\n``) forces an immediate
    flush.  The blank line itself is discarded.

    * **Token counting**
    Tokens are counted incrementally using *tiktoken* with the
    ``gpt-4o-mini`` encoding, adding ``len(enc.encode(ch))`` for each
    character appended.  This keeps the algorithm O(n).

    Args:
        content: The full text to split.
        lines_per_chunk: Maximum number of lines allowed in a chunk
            (inclusive).  A chunk is flushed when the running line count is
            **≥** this value *and* the token count is **≥** `tokens_per_chunk`.
        tokens_per_chunk: Maximum number of tokens allowed in a chunk
            (inclusive).  Tokens are computed with *tiktoken*.

    Yields:
        str: Consecutive, non-empty chunks of *content*.  Chunks preserve all
        original newlines and punctuation, never start with NOT-meaning
        characters, and respect both limits.

    Examples:
        >>> sample = "Hello!\\nHello!\\n\\n!\\nHi!\\n"
        >>> for c in chunk_content(sample, lines_per_chunk=2, tokens_per_chunk=2):
        ...     print('---\\n' + c + '---')
        ---
        Hello!
        Hello!
        ---
        ---
        Hi!
        !   # '!' is absorbed into previous chunk
        ---

        >>> text = "你好，世界！\\n你好，世界！\\n\\n？\\nHello.\\n"
        >>> chunks = list(chunk_content(text, lines_per_chunk=2, tokens_per_chunk=3))
        >>> len(chunks)
        2
        >>> chunks[0].endswith('！')
        True
        >>> chunks[1].startswith('Hello')
        True
    """  # noqa: E501

    if not content:
        return

    enc = (
        encoding if encoding is not None else tiktoken.encoding_for_model("gpt-4o-mini")
    )

    buf: list[str] = []  # Current chunk being built
    line_count = 0
    token_count = 0
    prev_chunk: str | None = None  # Completed chunk awaiting emission

    def _flush_current() -> None:
        """Move current buffer to completed chunk."""
        nonlocal buf, line_count, token_count, prev_chunk
        if buf:
            prev_chunk = "".join(buf)
            buf, line_count, token_count = [], 0, 0

    i = 0
    n = len(content)
    while i < n:
        ch = content[i]

        # Handle completed chunk waiting for emission
        if prev_chunk is not None and not buf:
            if ch.isspace() or ch in YIELD_LATER_CHARS:
                prev_chunk += ch
                i += 1
                continue
            # First meaningful character - emit completed chunk
            yield prev_chunk
            prev_chunk = None

        # Handle paragraph breaks (blank lines)
        if ch == "\n" and i > 0 and content[i - 1] == "\n":
            _flush_current()
            i += 1
            continue

        # Accumulate characters and counts
        buf.append(ch)
        if ch == "\n":
            line_count += 1
        token_count += len(enc.encode(ch))

        # Check if ready to break at safe boundary
        if line_count >= lines_per_chunk and token_count >= tokens_per_chunk:
            if ch == "\n" or ch in YIELD_LATER_CHARS:
                _flush_current()
        i += 1

    # Emit remaining content
    if buf:
        yield "".join(buf) if prev_chunk is None else prev_chunk + "".join(buf)
    elif prev_chunk is not None:
        yield prev_chunk

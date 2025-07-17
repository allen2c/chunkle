import typing
import unicodedata

import tiktoken

__version__ = "0.2.1"


def chunk(
    content: str,
    *,
    lines_per_chunk: int = 20,
    tokens_per_chunk: int = 500,
    encoding: tiktoken.Encoding | None = None,
) -> typing.Generator[str, None, None]:
    """
    Split text into chunks with minimum line and token requirements.

        **Algorithm:**
    1. Accumulate text until BOTH limits are met
    2. Flush at next break point: newline (best) > whitespace (good)
    3. If no break point found, force flush at 2x limits
    4. New chunks start with meaningful characters (non-whitespace/punctuation)

    **Args:**
        content: Text to split
        lines_per_chunk: Minimum lines per chunk (default: 20)
        tokens_per_chunk: Minimum tokens per chunk (default: 500)
        encoding: Custom tiktoken encoding (default: gpt-4o-mini)

    **Yields:**
        Text chunks preserving semantic boundaries when possible

    **Examples:**
        >>> list(chunk("Hello!\\nWorld!\\n", lines_per_chunk=1, tokens_per_chunk=2))
        ['Hello!\\n', 'World!\\n']
    """

    if not content:
        return

    enc = encoding or tiktoken.encoding_for_model("gpt-4o-mini")

    def _is_meaningful_char(ch: str) -> bool:
        if ch.isspace():
            return False

        # Get Unicode category
        cat = unicodedata.category(ch)

        # Important punctuation marks should be considered meaningful
        # to avoid breaking semantic units like quotes, parentheses, etc.
        if cat.startswith("P"):
            # Punctuation categories that should be treated as meaningful:
            # Ps (Open Punctuation) - (, [, {, 「, etc.
            # Pe (Close Punctuation) - ), ], }, 」, etc.
            # Pi (Initial Punctuation) - opening quotes
            # Pf (Final Punctuation) - closing quotes
            # Po (Other Punctuation) - sentence-ending punctuation like ., !, ?
            important_punct_cats = {"Ps", "Pe", "Pi", "Pf", "Po"}
            if cat in important_punct_cats:
                return True
            # Other punctuation categories (Pc, Pd) are less meaningful
            return False

        # All other non-space characters are meaningful
        return True

    def _is_good_break_point(content: str, pos: int) -> bool:
        """Check if this position is a good point to break between chunks."""
        if pos >= len(content):
            return True

        ch = content[pos]

        # Best break points: newlines (paragraph boundaries)
        if ch == "\n":
            return True

        # Other whitespace is good too, but prefer newlines
        if ch.isspace():
            return True

        return False

    buf: list[str] = []  # current chunk under construction
    line_count = 0
    token_count = 0
    pending_chunk: str | None = None  # completed chunk waiting to be yielded
    ready_to_flush = False  # flag to indicate we can flush at next good break point

    def _flush_current() -> None:
        """Move *buf* to *pending_chunk* and reset counters."""
        nonlocal buf, line_count, token_count, pending_chunk, ready_to_flush
        if buf:
            pending_chunk = "".join(buf)
            buf, line_count, token_count, ready_to_flush = [], 0, 0, False

    i = 0
    n = len(content)
    while i < n:
        ch = content[i]

        # 1️⃣ Handle a completed chunk that is waiting to be emitted
        if pending_chunk is not None and not buf:
            if not _is_meaningful_char(ch):
                # absorb punctuation/whitespace into *pending_chunk*
                pending_chunk += ch
                i += 1
                continue  # keep absorbing
            # first meaningful char → emit the previous chunk
            yield pending_chunk
            pending_chunk = None  # reset for next round

        # 2️⃣ Accumulate current character
        buf.append(ch)
        if ch == "\n":
            line_count += 1
        token_count += len(enc.encode(ch))

        # 3️⃣ Check if both limits are satisfied
        if line_count >= lines_per_chunk and token_count >= tokens_per_chunk:
            ready_to_flush = True

        # 4️⃣ Flush if ready and at a good break point
        if ready_to_flush and _is_good_break_point(content, i + 1):
            _flush_current()
        # Or if we've significantly exceeded limits, flush regardless
        elif line_count >= lines_per_chunk * 2 or token_count >= tokens_per_chunk * 2:
            _flush_current()

        i += 1

    # 5️⃣ Emit whatever is left
    if buf:
        yield "".join(buf) if pending_chunk is None else pending_chunk + "".join(buf)
    elif pending_chunk is not None:
        yield pending_chunk

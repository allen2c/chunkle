# chunkle/__init__.py
import logging
import typing

import tiktoken

__version__ = "0.3.0"

logger = logging.getLogger(__name__)


def chunk(
    content: str,
    *,
    lines_per_chunk: int = 20,
    tokens_per_chunk: int = 500,
    force_chunk_over_threshold_times: int = 2,
    encoding: tiktoken.Encoding | None = None,
) -> typing.Generator[str, None, None]:
    """Token-based chunking with dual thresholds and clean starts.
    Emit after both limits at next non-breaking token; trailing breaks merge.
    Force emit beyond multiplier only at whitespace token boundaries.
    """

    if not content:
        return
    if not (lines_per_chunk >= 1):
        raise ValueError("lines_per_chunk must be greater than or equal to 1")
    if not (tokens_per_chunk >= 1):
        raise ValueError("tokens_per_chunk must be greater than or equal to 1")
    if not (force_chunk_over_threshold_times >= 1):
        raise ValueError(
            "force_chunk_over_threshold_times must be greater than or equal to 1"
        )

    enc = encoding or tiktoken.encoding_for_model("gpt-4o-mini")
    # breaking_token_ids = set(get_breaking_token_ids(enc))

    buffer: typing.List[int] = []
    should_emit: bool = False
    current_lines: int = 1

    token_ids: typing.List[int] = enc.encode(content)
    token_texts: typing.List[str] = enc.decode_batch(
        [[token_id] for token_id in token_ids]
    )
    for token_id, token_text in zip(token_ids, token_texts):
        # A token is meaningful if it is not purely whitespace
        token_meaningful: bool = bool(token_text.strip())
        # A token is a breaking token if its decoded form starts/ends with a newline
        token_has_newline: bool = token_text.startswith("\n") or token_text.endswith(
            "\n"
        )

        # Emit when encounter meaningful characters but `should_emit` is True
        if token_meaningful and should_emit:
            yield enc.decode(buffer)
            buffer = []
            should_emit = False
            current_lines = 1

        buffer.append(token_id)

        if token_has_newline:
            current_lines += 1

        # Only after both conditions are met, we can check the condition of should emit
        if current_lines >= lines_per_chunk and len(buffer) >= tokens_per_chunk:

            # Should emit when encounter breaking token ids
            if token_has_newline:
                logger.debug(
                    f"Should emit chunk with lines: {current_lines}, "
                    + f"tokens: {len(buffer)}"
                )
                should_emit = True

            # Validate force emit condition: if the number of newlines is greater than the threshold times of the line per chunk  # noqa: E501
            elif current_lines >= lines_per_chunk * force_chunk_over_threshold_times:
                # Decode only when we need to inspect whitespace boundaries
                if (token_text[:1].isspace()) or (token_text[-1:].isspace()):
                    logger.debug(f"Force emit chunk due to lines: {current_lines}")
                    should_emit = True

            # Validate force emit condition: if the number of tokens is greater than the threshold times of the token per chunk  # noqa: E501
            elif len(buffer) >= tokens_per_chunk * force_chunk_over_threshold_times:
                # Decode only when we need to inspect whitespace boundaries
                if (token_text[:1].isspace()) or (token_text[-1:].isspace()):
                    logger.debug(f"Force emit chunk due to tokens: {len(buffer)}")
                    should_emit = True

            else:
                pass

    if buffer:
        yield enc.decode(buffer)

    return None

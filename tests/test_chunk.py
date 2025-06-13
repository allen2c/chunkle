import json
import pathlib
import typing
import unicodedata

import pytest
import tiktoken

from chunkle import chunk

SPECIAL_CHUNK_SEPARATOR = "<CHUNKLE_TESTCASE_SEPARATOR/>"

testcase_names = [
    "hello_world",
    "round_trip_preservation",
    "first_chunk_punctuation",
    "tight_limits",
    "huge_limits",
    "unicode_diversity",
    "unix_newlines",
    "windows_newlines",
    "no_final_newline",
]

RawContent: typing.TypeAlias = str
ChunkParams: typing.TypeAlias = typing.Dict[str, typing.Any]
ChunkContentWithSeparator: typing.TypeAlias = str


def _is_meaningful_char(ch: str) -> bool:
    """Helper function to check if a character is meaningful (same as chunk function)."""  # noqa: E501
    if ch.isspace():
        return False
    # Unicode punctuation categories (Pc, Pd, Pe, Pf, Pi, Po, Ps)
    return not unicodedata.category(ch).startswith("P")


def assert_chunk_invariants(raw: str, params: ChunkParams) -> None:
    """Assert key properties that all chunk operations must satisfy."""
    chunks = list(chunk(raw, **params))
    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    # 1. Round-trip preservation
    assert "".join(chunks) == raw, "Chunks must preserve original content exactly"

    # 2. Budget law - all chunks except the last must meet both thresholds
    for i, c in enumerate(chunks[:-1]):  # all but last
        line_count = c.count("\n")
        token_count = len(enc.encode(c))
        lines_msg = (
            f"Chunk {i} has {line_count} lines, "
            f"expected >= {params['lines_per_chunk']}"
        )
        tokens_msg = (
            f"Chunk {i} has {token_count} tokens, "
            f"expected >= {params['tokens_per_chunk']}"
        )
        assert line_count >= params["lines_per_chunk"], lines_msg
        assert token_count >= params["tokens_per_chunk"], tokens_msg

    # 3. First-chunk caveat - chunks after the first must start with meaningful char
    if len(chunks) > 1:
        for i, c in enumerate(chunks[1:], 1):  # skip first chunk
            if c:  # non-empty chunk
                msg = (
                    f"Chunk {i} starts with non-meaningful character "
                    f"'{c[0]}' (ord={ord(c[0])})"
                )
                assert _is_meaningful_char(c[0]), msg


def _prepare_testcases() -> (
    typing.Generator[
        tuple[RawContent, ChunkParams, ChunkContentWithSeparator], None, None
    ]
):
    for testcase_name in testcase_names:
        testcases_dir = pathlib.Path(__file__).parent.joinpath("testcases").resolve()
        if not testcases_dir.exists():
            raise FileNotFoundError(f"Testcases directory not found: {testcases_dir}")

        raw_content = testcases_dir.joinpath(f"{testcase_name}.txt").read_text()
        chunk_params = json.loads(
            testcases_dir.joinpath(f"{testcase_name}.json").read_text()
        )
        chunks_content_with_separator = testcases_dir.joinpath(
            f"{testcase_name}_chunks.txt"
        ).read_text()

        yield raw_content, chunk_params, chunks_content_with_separator


@pytest.mark.parametrize(
    "raw_content, chunk_params, chunks_content_with_separator",
    list(_prepare_testcases()),
)
def test_chunk(
    raw_content: RawContent,
    chunk_params: ChunkParams,
    chunks_content_with_separator: ChunkContentWithSeparator,
):
    assert chunks_content_with_separator == SPECIAL_CHUNK_SEPARATOR.join(
        chunk(raw_content, **chunk_params)
    )


@pytest.mark.parametrize(
    "raw_content, chunk_params, chunks_content_with_separator",
    list(_prepare_testcases()),
)
def test_chunk_invariants(
    raw_content: RawContent,
    chunk_params: ChunkParams,
    chunks_content_with_separator: ChunkContentWithSeparator,
):
    """Test that all chunking operations satisfy the key invariants."""
    assert_chunk_invariants(raw_content, chunk_params)

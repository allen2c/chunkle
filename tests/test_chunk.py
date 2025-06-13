import json
import pathlib
import typing

import pytest

from chunkle import chunk

SPECIAL_CHUNK_SEPARATOR = "<CHUNKLE_TESTCASE_SEPARATOR/>"

testcase_names = [
    "hello_world",
    "large_limits",
    "token_heavy",
    "multilingual_complex",
    "document_format",
    "edge_cases",
]

RawContent: typing.TypeAlias = str
ChunkParams: typing.TypeAlias = typing.Dict[str, typing.Any]
ChunkContentWithSeparator: typing.TypeAlias = str


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

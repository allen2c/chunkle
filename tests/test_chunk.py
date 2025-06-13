import json
import pytest
import pathlib
import typing

from chunkle import chunk

SPECIAL_CHUNK_SEPARATOR = "---CHUNKLE-TESTCASE-SEPARATOR---"

testcase_names = [
    "hello_world",
]

RawContent: typing.TypeAlias = str
ChunkContent: typing.TypeAlias = str
ChunkParams: typing.TypeAlias = typing.Dict[str, typing.Any]


def _prepare_testcases() -> (
    typing.Generator[
        tuple[RawContent, ChunkParams, typing.List[ChunkContent]], None, None
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
        chunk_content = testcases_dir.joinpath(
            f"{testcase_name}_chunks.txt"
        ).read_text()
        chunks = chunk_content.split(SPECIAL_CHUNK_SEPARATOR)

        yield raw_content, chunk_params, chunks


@pytest.mark.parametrize(
    "raw_content, chunk_params, chunks", list(_prepare_testcases())
)
def test_chunk(
    raw_content: RawContent,
    chunk_params: ChunkParams,
    chunks: typing.List[ChunkContent],
):
    assert chunks == list(chunk(raw_content, **chunk_params))

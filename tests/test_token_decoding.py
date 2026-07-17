import typing

import pytest
import tiktoken

from chunkle import chunk

# chunk() decodes each token with decode_single_token_bytes. The obvious
# alternative, decode_batch([[t] for t in ids]), submits one thread-pool future
# per token and costs 65x more for identical output.

ENC = tiktoken.encoding_for_model("gpt-4o-mini")


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "The quick brown fox jumps over the lazy dog. 1234567890", id="ascii"
        ),
        # Multi-byte characters land on token boundaries, so a single token may
        # hold a partial UTF-8 sequence that only errors="replace" can decode.
        pytest.param(
            "台灣的氣候屬於亞熱帶與熱帶交界，夏季炎熱潮濕，冬季溫和。", id="zh_tw"
        ),
        pytest.param("日本語テキストの分割テスト", id="japanese"),
        pytest.param("👨‍👩‍👧‍👦🇹🇼🎉", id="emoji"),
        pytest.param("\n\n\t  \n", id="whitespace_only"),
        pytest.param("first line\r\nsecond line\n\nthird line", id="mixed_newlines"),
    ],
)
def test_decode_single_token_matches_decode_batch(text: str):
    """Per-token decoding must stay byte-identical to tiktoken's own decoding."""
    token_ids = ENC.encode(text)
    assert token_ids, "test corpus must produce at least one token"

    via_single_token = [
        ENC.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
        for token_id in token_ids
    ]

    assert via_single_token == ENC.decode_batch([[t] for t in token_ids])


@pytest.mark.parametrize("special_text", sorted(ENC.special_tokens_set))
def test_special_tokens_decode_rather_than_raise(special_text: str):
    """Special tokens decode to their text form instead of raising."""
    token_id = ENC.encode_single_token(special_text)

    decoded = ENC.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")

    assert decoded == special_text


def test_chunk_does_not_use_decode_batch(monkeypatch: pytest.MonkeyPatch):
    """chunk() must decode per token, never via the thread-pooled decode_batch.

    The equivalence tests above hold for either implementation, so this is what
    actually keeps the fast path from regressing.
    """

    def forbidden(*_args: typing.Any, **_kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError(
            "chunk() called decode_batch, which submits one thread-pool future "
            "per token; decode per token instead"
        )

    monkeypatch.setattr(tiktoken.Encoding, "decode_batch", forbidden)

    text = "台灣的氣候屬於亞熱帶與熱帶交界。\n" * 20
    chunks = list(chunk(text, encoding=ENC, lines_per_chunk=5, tokens_per_chunk=20))

    assert "".join(chunks) == text

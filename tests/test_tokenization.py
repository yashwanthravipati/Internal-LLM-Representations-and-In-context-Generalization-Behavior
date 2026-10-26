import pytest

from icl_geometry.tasks.templates import TEMPLATES
from icl_geometry.tasks.tokenization import item_tokens, single_token_id

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    except OSError:
        pytest.skip("tokenizer not cached")


def test_template_spans():
    for t in TEMPLATES.values():
        text, (s, e) = t.render([("dog", " foo"), ("oak", " bar")], "wolf")
        assert text[s:e] == "wolf"
        assert " foo" in text and " bar" in text
        assert not text.endswith(" ")


@pytest.mark.slow
def test_item_tokens_in_context(tok):
    arrow = item_tokens(tok, "wolf", "arrow")
    io_ = item_tokens(tok, "wolf", "io")
    assert arrow.clean_boundary and io_.clean_boundary
    assert arrow.suffix_ok and io_.suffix_ok
    assert io_.n_tokens == 1  # " wolf" with leading space
    assert item_tokens(tok, "37", "io").n_tokens == 2  # digits are split


@pytest.mark.slow
def test_single_token_id(tok):
    assert single_token_id(tok, "foo") is not None
    assert single_token_id(tok, "landlocked") is None

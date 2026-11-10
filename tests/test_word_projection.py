import numpy as np
import pytest

from ir_hw4 import word_projection


@pytest.mark.parametrize("method", word_projection.PROJECTION_METHODS)
def test_saved_projection_is_reused_only_for_the_same_model(tmp_path, monkeypatch, method):
    monkeypatch.setattr(word_projection, "PROJECTION_CACHE", tmp_path)
    vectors = np.random.default_rng(12).normal(size=(7, 4))
    vectors[-1] = 0
    expected = word_projection.get_vocabulary_projection("model-a", "cbow", method, vectors)
    assert np.isfinite(expected["xy"][:-1]).all()
    assert np.isnan(expected["xy"][-1]).all()

    def unexpected_compute(*_):
        raise RuntimeError("projection recalculated")

    monkeypatch.setattr(word_projection, "compute_vocabulary_projection", unexpected_compute)
    cached = word_projection.get_vocabulary_projection("model-a", "cbow", method, vectors)
    for field, value in expected.items():
        np.testing.assert_array_equal(cached[field], value)
    with pytest.raises(RuntimeError, match="projection recalculated"):
        word_projection.get_vocabulary_projection("model-b", "cbow", method, vectors)


def test_unreadable_projection_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(word_projection, "PROJECTION_CACHE", tmp_path)
    vectors = np.random.default_rng(14).normal(size=(6, 4))
    expected = word_projection.get_vocabulary_projection("model-a", "skipgram", "PCA", vectors)
    path = next(tmp_path.glob("*.npz"))
    path.write_bytes(b"incomplete cache")
    actual = word_projection.get_vocabulary_projection("model-a", "skipgram", "PCA", vectors)
    np.testing.assert_array_equal(actual["xy"], expected["xy"])
    assert word_projection.read_cached_projection(path, vectors, "PCA") is not None

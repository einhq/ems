"""`embed_texts` in the model worker, with the embedder stubbed: roles, families and the token limit."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import inference_ops as ops  # noqa: E402
import model_registry as registry  # noqa: E402


class Vectors(list):
    """What `encode(convert_to_numpy=True)` answers, as far as `embed_texts` reads it."""

    def tolist(self):
        return list(self)


class FakeEmbedder:
    prompts = {"query": "Query: ", "document": "Document: "}

    def __init__(self):
        self.calls = []

    def tokenizer(self, texts, add_special_tokens, truncation):
        # One token a word, prompt included, so a limit is easy to reach.
        return {"input_ids": [text.split() for text in texts]}

    def encode(self, texts, prompt_name, **kwargs):
        self.calls.append((list(texts), prompt_name))
        return Vectors([1.0, 0.0] for _ in texts)


@pytest.fixture
def embedder(monkeypatch):
    fake = FakeEmbedder()
    loaded = []
    monkeypatch.setattr(registry, "text_embedder", lambda family, device: loaded.append((family, device)) or fake)
    monkeypatch.setattr(registry, "text_embed_device", lambda: "cuda")
    monkeypatch.setattr(registry, "text_embed_query_device", lambda: "cpu")
    monkeypatch.setattr(registry, "run_on_model_thread", lambda family, fn: fn())
    monkeypatch.setattr(registry, "TEXT_EMBED_MAX_TOKENS", 4)
    fake.loaded = loaded
    return fake


def test_a_query_runs_in_its_own_family_on_the_query_device(embedder):
    result = ops.embed_texts(["en segelbåt"], "query")
    assert result == {"embeddings": [[1.0, 0.0]], "model": registry.TEXT_EMBED_MODEL.split("/")[-1]}
    assert embedder.loaded == [("text-query", "cpu")]
    assert embedder.calls == [(["en segelbåt"], "query")]


def test_documents_run_in_the_document_family(embedder):
    ops.embed_texts(["a", "b"], "document")
    assert embedder.loaded == [("text-embed", "cuda")]
    assert embedder.calls == [(["a", "b"], "document")]


def test_documents_someone_waits_on_run_in_the_query_family_as_documents(embedder):
    ops.embed_texts(["a"], "document", None, "query")
    assert embedder.loaded == [("text-query", "cpu")]
    assert embedder.calls == [(["a"], "document")]


def test_a_text_over_the_limit_is_refused_not_cut(embedder):
    # "Document: " is one token here, so four words is five.
    with pytest.raises(ValueError, match="text 1 is 5 tokens, over the 4-token limit"):
        ops.embed_texts(["tre små ord", "fyra ord är många"], "document")
    assert embedder.calls == []


def test_an_unknown_role_is_refused(embedder):
    with pytest.raises(ValueError, match="role"):
        ops.embed_texts(["a"], "passage")


def test_documents_are_placed_on_the_model_thread_not_before_it(embedder, monkeypatch):
    # Placing outside the family's thread and lock could move the family under a request that
    # is using it (ein/core#736).
    placed, threads = [], []
    monkeypatch.setattr(registry, "text_embed_device", lambda: placed.append("placed") or "cuda")
    monkeypatch.setattr(registry, "run_on_model_thread", lambda family, fn: threads.append(family) or (placed.append("thread") or fn()))
    ops.embed_texts(["a"], "document")
    assert placed == ["thread", "placed"]


def test_documents_the_card_takes_leave_no_copy_on_the_cpu(embedder, monkeypatch):
    monkeypatch.setattr(registry, "_cache", {"text-embed:cpu": object()})
    monkeypatch.setattr(registry, "free_vram_mb", lambda: 9000.0)
    monkeypatch.setattr(registry, "cleanup_torch", lambda: None)
    ops.embed_texts(["a"], "document", device="card")
    assert embedder.loaded == [("text-embed", "cuda")]
    assert "text-embed:cpu" not in registry._cache

"""SSE streaming endpoint tests (Product Roadmap P3). Keyless (EchoClient streams words)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.core import config as config_mod
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.main import app
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore

client = TestClient(app)


@pytest.fixture()
def _wired(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "assistant_contract_v1", True, raising=False)
    echo = EchoClient()
    app.state.llm_router = LLMRouter(
        clients={TASK_REASONING: echo, TASK_SYNTHESIS: echo, TASK_FALLBACK: echo},
        fallback=echo,
    )
    app.state.rag = {
        "embedding_provider": LocalHashEmbedding(dims=16),
        "vector_store": InMemoryVectorStore(),
        "llm_client": echo,
    }
    yield


def _frames(body: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_stream_emits_sse_deltas_then_a_typed_envelope(_wired):
    r = client.post("/api/v1/assistant/stream", json={"query": "How do I ship a box?"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    frames = _frames(r.text)
    deltas = [f for f in frames if "delta" in f]
    done = [f for f in frames if f.get("done")]

    assert len(deltas) > 1  # actually streamed, not one blob
    assert len(done) == 1
    env = done[0]["assistant"]
    assert env["type"] == "answer" and env["schema_version"] == "1"
    # the final envelope's message equals the concatenated stream (lossless)
    assert env["message"] == "".join(f["delta"] for f in deltas)
    assert env["result"]["type"] == "policy_answer"


def test_stream_404_when_contract_disabled(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "assistant_contract_v1", False, raising=False)
    r = client.post("/api/v1/assistant/stream", json={"query": "hi"})
    assert r.status_code == 404


def test_stream_503_when_pipeline_uninitialized(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "assistant_contract_v1", True, raising=False)
    app.state.llm_router = None
    app.state.rag = None
    r = client.post("/api/v1/assistant/stream", json={"query": "hi"})
    assert r.status_code == 503


def test_stream_validates_empty_query(_wired):
    r = client.post("/api/v1/assistant/stream", json={"query": ""})
    assert r.status_code == 422

from unittest.mock import Mock

import pytest
from api import main
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "global_state", {"rag_chain": None})
    with TestClient(main.app) as client:
        yield client


def test_health(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "FinSight API is running smoothly"


def test_ask_contract(client):
    chain = Mock()
    chain.invoke.side_effect = lambda payload: "Yanıt: " + payload["input"]
    main.global_state["rag_chain"] = chain
    response = client.post("/ask/", json={"question": "Gelir?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Yanıt: Gelir?"}


@pytest.mark.parametrize(
    "body, status", [({}, 422), ({"question": ""}, 400), ({"question": "  "}, 400)]
)
def test_invalid_question(client, body, status):
    assert client.post("/ask/", json=body).status_code == status


def test_requires_report(client):
    response = client.post("/ask/", json={"question": "Gelir?"})
    assert response.status_code == 400
    assert "upload" in response.json()["detail"]


def test_wrong_upload_type(client):
    response = client.post("/upload/", files={"file": ("notes.txt", b"text", "text/plain")})
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_rag_failure(client):
    chain = Mock()
    chain.invoke.side_effect = RuntimeError("provider unavailable")
    main.global_state["rag_chain"] = chain
    response = client.post("/ask/", json={"question": "Gelir?"})
    assert response.status_code == 500
    assert response.json()["detail"] == "LLM Error: provider unavailable"


def test_upload_sets_active_chain_and_removes_temporary_pdf(client, tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    processor = Mock()
    processor.process_pdf.side_effect = lambda path: [Path(path).read_bytes().decode()]
    monkeypatch.setattr(main, "FinancialPDFProcessor", lambda: processor)
    manager = Mock()
    database = Mock()
    def store(chunks, name):
        database.as_retriever.return_value = chunks
        return database

    manager.create_and_store_embeddings.side_effect = store
    monkeypatch.setattr(main, "VectorStoreManager", lambda: manager)
    service = Mock()
    service.create_rag_chain.side_effect = lambda retriever: Mock(
        invoke=lambda query: retriever[0] + ": " + query["input"]
    )
    monkeypatch.setattr(main, "LLMService", lambda: service)
    response = client.post("/upload/", files={"file": ("sample.pdf", b"report content")})
    assert response.status_code == 200
    assert response.json()["filename"] == "sample.pdf"
    assert not (tmp_path / "data" / "temp_uploads" / "sample.pdf").exists()
    assert client.post("/ask/", json={"question": "Q"}).json() == {"answer": "report content: Q"}

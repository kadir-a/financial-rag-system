from contextlib import ExitStack
from shutil import rmtree

from langchain_core.embeddings import Embeddings

from app.services import vector_store


class TopicEmbeddings(Embeddings):
    def embed_query(self, text):
        return [float(word in text) for word in ("revenue", "debt", "cash")]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def test_report_switch_and_top_k(tmp_path, monkeypatch):
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.setattr(vector_store, "OpenAIEmbeddings", lambda **kw: TopicEmbeddings())
    root = tmp_path / "vaults"
    manager = vector_store.VectorStoreManager(root)
    for report in ("report_alpha", "report_beta"):
        chunks = [f"{report} {topic}" for topic in ("cash", "debt", "revenue")]
        with ExitStack() as clients:
            stored = manager.create_and_store_embeddings(chunks, report)
            clients.callback(stored._client.close)
            reopened = manager.get_vector_db(report)
            clients.callback(reopened._client.close)
            assert reopened._collection.name == report
            for k in (1, 2, 3):
                docs = reopened.as_retriever(search_kwargs={"k": k}).invoke("revenue")
                texts = [doc.page_content for doc in docs]
                assert len(texts) == k
                assert texts[0] == f"{report} revenue"
                assert set(texts) <= set(chunks)
    rmtree(root)
    assert not root.exists()

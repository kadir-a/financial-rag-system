from types import SimpleNamespace
from unittest.mock import Mock

from app.services import llm_service


def test_rewrite_only_with_history(monkeypatch):
    messages = []

    def respond(prompt):
        messages.append(prompt[0].content)
        return SimpleNamespace(content="2025 gelir" if "Clear Search Query:" in messages[-1]
                               else "42 TL")

    llm = Mock()
    llm.invoke.side_effect = respond
    monkeypatch.setattr(llm_service, "ChatOpenAI", lambda **kw: llm)
    retriever = Mock()
    retriever.invoke.side_effect = lambda query: [SimpleNamespace(page_content=f"{query}: 42")]
    chain = llm_service.LLMService().create_rag_chain(retriever)
    trace = chain.invoke_with_trace({"input": "Gelir?", "chat_history": []})
    assert trace["contexts"] == ["Gelir?: 42"]
    assert len(messages) == 1
    history = [["user", "2024 gelir?"], ["assistant", "30 TL"]]
    trace = chain.invoke_with_trace({"input": "Peki 2025?", "chat_history": history})
    assert "User: 2024 gelir?\nAssistant: 30 TL" in messages[1]
    assert "Latest Question: Peki 2025?" in messages[1]
    assert trace["contexts"] == ["2025 gelir: 42"]
    assert "Question: Peki 2025?" in messages[2]
    assert trace["answer"] == "42 TL"
    assert trace["chat_usage"] == {"rewrite": None, "generation": None}

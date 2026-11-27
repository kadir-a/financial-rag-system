import hashlib
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

# --- CORE SERVICES INTEGRATION ---
from services.ingestion import FinancialPDFProcessor
from services.llm_service import LLMService
from services.vector_store import VectorStoreManager

from app.db.database import best_effort
from app.db.repository import record_turn, register_report, token_counts

app = FastAPI(title="FinSight AI Backend API", version="1.0")

# Global state memory (Temporarily stores the active collection name and RAG chain)
global_state = {"current_collection_name": None, "rag_chain": None}


# Payload Formats
class QueryRequest(BaseModel):
    session_id: UUID | None = None
    question: str
    chat_history: list = []  # Includes previous conversation context


@app.get("/")
def health_check():
    return {"status": "FinSight API is running smoothly", "timestamp": time.time()}


@app.post("/upload/")
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    company: str | None = Form(None),
    period: str | None = Form(None),
):
    """Ingests a PDF, parses it, creates vector embeddings, and prepares the RAG chain."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # 1. Temporarily store the file on disk (Required for LlamaParse processing)
    temp_dir = "./data/temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    filename = Path(file.filename.replace("\\", "/")).name
    with NamedTemporaryFile(dir=temp_dir, suffix=".pdf", delete=False) as buffer:
        file_path = buffer.name
    try:
        digest = hashlib.sha256()
        with open(file_path, "wb") as output:
            while block := file.file.read(1024 * 1024):
                digest.update(block)
                output.write(block)
        try:
            page_count = len(PdfReader(file_path).pages)
        except Exception:
            page_count = None
        # 2. Initialize Core Engines
        processor = FinancialPDFProcessor()
        vector_manager = VectorStoreManager()
        llm_service = LLMService()

        # 3. Document Ingestion
        print(f"API: Processing {file.filename}...")
        chunks = processor.process_pdf(file_path)

        # 4. Generate a unique vault for each upload (Prevents data leakage)
        collection_name = f"vault_{uuid4().hex}"

        # 5. Convert to Embeddings (Vector Store)
        vector_db = vector_manager.create_and_store_embeddings(chunks, collection_name)

        # 6. Configure Retriever and establish the Custom RAG pipeline
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        chain = llm_service.create_rag_chain(retriever)
        metadata = dict(
            filename=filename,
            sha256=digest.hexdigest(),
            company=company,
            period=period,
            page_count=page_count,
            chroma_collection=collection_name,
        )
        best_effort(register_report, metadata)
        global_state.update(
            rag_chain=chain, current_collection_name=collection_name, report_metadata=metadata
        )
        response.delete_cookie("finsight_session")
        return {
            "filename": file.filename,
            "status": "success",
            "collection_id": collection_name,
            "message": "Document successfully ingested and stored in the vector vault.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Processing Error: {str(e)}")

    finally:
        Path(file_path).unlink(missing_ok=True)


@app.post("/ask/")
async def ask_ai(query: QueryRequest, request: Request, response: Response):
    """Receives the user query, searches the active vault, and returns the generated answer."""
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not global_state["rag_chain"]:
        raise HTTPException(status_code=400, detail="You must upload a financial report first.")

    try:
        # Format payload for the RAG engine
        payload = {"input": query.question, "chat_history": query.chat_history}

        # Trigger the CustomRAG chain
        chain = global_state["rag_chain"]
        metadata = global_state.get("report_metadata")
        start = time.perf_counter()
        if metadata and hasattr(type(chain), "invoke_with_trace"):
            trace = chain.invoke_with_trace(payload)
            answer = trace["answer"]
        else:
            answer = chain.invoke(payload)
            trace = {}
        if metadata:
            session_id = query.session_id
            if session_id is None:
                cookie = request.cookies.get("finsight_session", "")
                prefix = metadata["sha256"] + "."
                try:
                    session_id = (
                        UUID(cookie.removeprefix(prefix)) if cookie.startswith(prefix) else None
                    )
                except ValueError:
                    session_id = None
                session_id = session_id or uuid4()
            metrics = dict(
                latency_ms=(time.perf_counter() - start) * 1000,
                rewritten_query=trace.get("search_query") if query.chat_history else None,
                **token_counts({**trace, "had_history": bool(query.chat_history)}),
            )
            best_effort(record_turn, metadata, session_id, query.question, answer, metrics)
            response.set_cookie(
                "finsight_session",
                f"{metadata['sha256']}.{session_id}",
                httponly=True,
                samesite="strict",
            )
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

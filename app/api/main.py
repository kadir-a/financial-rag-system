from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import time
import os
import shutil
import uuid

# --- CORE SERVICES INTEGRATION ---
from services.ingestion import FinancialPDFProcessor
from services.vector_store import VectorStoreManager
from services.llm_service import LLMService

app = FastAPI(title="FinSight AI Backend API", version="1.0")

# Global state memory (Temporarily stores the active collection name and RAG chain)
global_state = {
    "current_collection_name": None,
    "rag_chain": None
}

# Payload Formats
class QueryRequest(BaseModel):
    question: str
    chat_history: list = []  # Includes previous conversation context

@app.get("/")
def health_check():
    return {"status": "FinSight API is running smoothly", "timestamp": time.time()}

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """Ingests a PDF, parses it, creates vector embeddings, and prepares the RAG chain."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    # 1. Temporarily store the file on disk (Required for LlamaParse processing)
    temp_dir = "./data/temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Initialize Core Engines
        processor = FinancialPDFProcessor()
        vector_manager = VectorStoreManager()
        llm_service = LLMService()
        
        # 3. Document Ingestion
        print(f"API: Processing {file.filename}...")
        chunks = processor.process_pdf(file_path)
        
        # 4. Generate a unique vault for each upload (Prevents data leakage)
        collection_name = f"vault_{int(time.time())}"
        global_state["current_collection_name"] = collection_name
        
        # 5. Convert to Embeddings (Vector Store)
        vector_db = vector_manager.create_and_store_embeddings(chunks, collection_name)
        
        # 6. Configure Retriever and establish the Custom RAG pipeline
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        global_state["rag_chain"] = llm_service.create_rag_chain(retriever)
        
        # 7. Cleanup temporary file
        os.remove(file_path)
        
        return {
            "filename": file.filename, 
            "status": "success", 
            "collection_id": collection_name,
            "message": "Document successfully ingested and stored in the vector vault."
        }
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"PDF Processing Error: {str(e)}")

@app.post("/ask/")
async def ask_ai(query: QueryRequest):
    """Receives the user query, searches the active vault, and returns the generated answer."""
    if not query.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    if not global_state["rag_chain"]:
        raise HTTPException(status_code=400, detail="You must upload a financial report first.")
    
    try:
        # Format payload for the RAG engine
        payload = {
            "input": query.question,
            "chat_history": query.chat_history
        }
        
        # Trigger the CustomRAG chain
        answer = global_state["rag_chain"].invoke(payload)
        
        return {"answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
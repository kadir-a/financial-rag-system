from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import time
import os
import shutil
import uuid

# --- SİSTEMİN BEYNİNİ (SERVICES) API'YE BAĞLIYORUZ ---
from services.ingestion import FinancialPDFProcessor
from services.vector_store import VectorStoreManager
from services.llm_service import LLMService

app = FastAPI(title="FinSight AI Backend API", version="1.0")

# Global hafıza (Geçici olarak aktif koleksiyon adını ve RAG chain'i tutar)
global_state = {
    "current_collection_name": None,
    "rag_chain": None
}

# Payload Formatları
class QueryRequest(BaseModel):
    question: str
    chat_history: list = []  # Soru ile birlikte geçmiş konuşmalar da gelecek

@app.get("/")
def health_check():
    return {"status": "FinSight API is running smoothly", "timestamp": time.time()}

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """PDF'i alır, okur, vektöre çevirir ve sistemi soruya hazır hale getirir."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyaları kabul edilmektedir.")
    
    # 1. Dosyayı geçici olarak diskte tut (LlamaParse'ın okuyabilmesi için)
    temp_dir = "./data/temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Motorları Çalıştır
        processor = FinancialPDFProcessor()
        vector_manager = VectorStoreManager()
        llm_service = LLMService()
        
        # 3. PDF'i İşle (Ingestion)
        print(f"API: {file.filename} işleniyor...")
        chunks = processor.process_pdf(file_path)
        
        # 4. Her yüklemede benzersiz bir kasa aç (Veri Sızıntısını Önlemek İçin)
        collection_name = f"vault_{int(time.time())}"
        global_state["current_collection_name"] = collection_name
        
        # 5. Vektörlere Çevir (Vector Store)
        vector_db = vector_manager.create_and_store_embeddings(chunks, collection_name)
        
        # 6. Retriever (Arama Motoru) Kur ve RAG Borusunu (CustomRAG) Hazırla
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        global_state["rag_chain"] = llm_service.create_rag_chain(retriever)
        
        # 7. Geçici dosyayı temizle
        os.remove(file_path)
        
        return {
            "filename": file.filename, 
            "status": "success", 
            "collection_id": collection_name,
            "message": "Doküman başarıyla işlendi ve vektör kasasına aktarıldı."
        }
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"PDF İşleme Hatası: {str(e)}")

@app.post("/ask/")
async def ask_ai(query: QueryRequest):
    """Kullanıcının sorusunu alır, aktif kasada arar ve cevabı döner."""
    if not query.question:
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")
        
    if not global_state["rag_chain"]:
        raise HTTPException(status_code=400, detail="Önce bir finansal rapor yüklemelisiniz.")
    
    try:
        # Payload'u RAG motorunun beklediği formata sokuyoruz
        payload = {
            "input": query.question,
            "chat_history": query.chat_history
        }
        
        # Manuel vites CustomRAG'i ateşle
        answer = global_state["rag_chain"].invoke(payload)
        
        return {"answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Hatası: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
import os
from app.services.ingestion import FinancialPDFProcessor
from app.services.vector_store import VectorStoreManager
from app.services.llm_service import LLMService

def main():
    print("Operasyon Başlıyor: Tam RAG Sistemi...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "data", "source_pdfs", "arcelik_2025_fr.pdf")
    
    # 1. Okuma (Zaten kayıtlı olduğu için hızlı geçecek)
    processor = FinancialPDFProcessor()
    chunks = processor.process_pdf(pdf_path)
    
    # 2. Hafıza (ChromaDB)
    vector_manager = VectorStoreManager()
    vector_db = vector_manager.create_and_store_embeddings(chunks, "arcelik_report")
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # 3. Beyin (GPT-4o-mini)
    print("\nYapay Zeka Beyni Aktive Ediliyor...")
    llm_service = LLMService()
    rag_chain = llm_service.create_rag_chain(retriever)
    
    # 4. Nihai Test
    print("\n--- FİNANSAL ANALİZ SONUCU ---")
    query = "Şirketin net dönem zararı ne kadar?"
    print(f"Sen: {query}\n")
    
    # Sistemi tetikle
    response = rag_chain.invoke(query)
    
    print(f"Uzman Analist: {response}")

if __name__ == "__main__":
    main()
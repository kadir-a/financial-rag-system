import os
from app.services.ingestion import FinancialPDFProcessor
from app.services.vector_store import VectorStoreManager
from app.services.llm_service import LLMService

def main():
    print("Starting Operation: Full RAG System Validation...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "data", "source_pdfs", "arcelik_2025_fr.pdf")
    
    # 1. Ingestion
    processor = FinancialPDFProcessor()
    chunks = processor.process_pdf(pdf_path)
    
    # 2. Memory (ChromaDB) - Updated to the new 1536 dimension vault
    vector_manager = VectorStoreManager()
    vector_db = vector_manager.create_and_store_embeddings(chunks, "test_report_openai_1536")
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # 3. Brain (GPT-4o-mini via CustomRAG)
    print("\nActivating AI Engine...")
    llm_service = LLMService()
    rag_chain = llm_service.create_rag_chain(retriever)
    
    # 4. Final Execution
    print("\n--- FINANCIAL ANALYSIS RESULT ---")
    query = "What is the net loss for the period?"
    print(f"User: {query}\n")
    
    # Trigger the system with the correct payload structure (dict)
    response = rag_chain.invoke({
        "input": query,
        "chat_history": []
    })
    
    print(f"Expert Analyst: {response}")

if __name__ == "__main__":
    main()
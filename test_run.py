import os
import time
from app.services.ingestion import FinancialPDFProcessor
from app.services.vector_store import VectorStoreManager
from app.services.llm_service import LLMService

def main():
    print("🚀Starting Operation: Headless RAG System Validation...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Placeholder for headless testing. You can place any PDF named 'sample_report.pdf' here to test the engine directly.
    pdf_path = os.path.join(current_dir, "data", "source_pdfs", "sample_report.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"⚠️ System Halt: Test PDF not found at {pdf_path}.")
        print("Please place a 'sample_report.pdf' in the directory to run the headless integration test.")
        return

    # 1. Ingestion Engine
    processor = FinancialPDFProcessor()
    chunks = processor.process_pdf(pdf_path)
    
    # 2. Memory Vault (ChromaDB) - Updated with unique timestamp and expanded field of view
    vector_manager = VectorStoreManager()
    unique_test_vault = f"test_vault_{int(time.time())}"
    vector_db = vector_manager.create_and_store_embeddings(chunks, unique_test_vault)
    
    # Field of view expanded to 8 to capture full Markdown tables
    retriever = vector_db.as_retriever(search_kwargs={"k": 8})
    
    # 3. Core Brain (GPT-4o-mini via CustomRAG)
    print("\n🧠 Activating AI Engine...")
    llm_service = LLMService()
    rag_chain = llm_service.create_rag_chain(retriever)
    
    # 4. Final Execution Strategy
    print("\n--- FINANCIAL ANALYSIS RESULT ---")
    query = "What is the total assets for the period?" 
    print(f"User: {query}\n")
    
    # Trigger the system with the correct payload structure (dict) and empty history for clean test
    response = rag_chain.invoke({
        "input": query,
        "chat_history": []
    })
    
    print(f"Expert Analyst: {response}")

if __name__ == "__main__":
    main()
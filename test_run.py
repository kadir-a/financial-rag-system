import os
from app.services.ingestion import FinancialPDFProcessor
from app.services.vector_store import VectorStoreManager

def main():
    print("Starting Operation...")
    
    # 1. File Path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "data", "source_pdfs", "arcelik_2025_fr.pdf")
    
    # 2. Extract and Chunk Data
    processor = FinancialPDFProcessor()
    chunks = processor.process_pdf(pdf_path)
    
    # 3. Convert to Vectors and Store
    vector_manager = VectorStoreManager()
    vector_db = vector_manager.create_and_store_embeddings(chunks, "arcelik_report")
    
    # 4. Test System Memory (Similarity Search)
    print("\n--- QUICK SEARCH TEST ---")
    query = "Şirketin net dönem zararı ne kadar?"
    print(f"Query: {query}")
    
    results = vector_db.similarity_search(query, k=1)
    
    print("\nClosest Match Found:")
    print(results[0].page_content)

if __name__ == "__main__":
    main()
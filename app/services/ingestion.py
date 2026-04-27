import fitz  # PyMuPDF
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class FinancialPDFProcessor:
    def __init__(self, chunk_size=1200, chunk_overlap=250):
        # Higher overlap is maintained to prevent context loss in financial data
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def extract_hybrid_text(self, pdf_path: str) -> str:
        """Extracts text using fitz and tables using pdfplumber."""
        full_text = ""
        
        print("Analyzing text...")
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += f"\n--- PAGE {page_num + 1} TEXT ---\n"
                full_text += page.get_text("text") + "\n"
        except Exception as e:
            print(f"PyMuPDF Error: {e}")

        print("Scanning financial tables...")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    extracted_tables = page.extract_tables()
                    if extracted_tables:  # Proceed only if a table is found
                        for i, table in enumerate(extracted_tables):
                            full_text += f"\n--- PAGE {page_num + 1} TABLE {i + 1} ---\n"
                            for row in table:
                                # Clean whitespaces and None values in cells
                                clean_row = [str(cell).replace('\n', ' ') if cell else "-" for cell in row]
                                full_text += " | ".join(clean_row) + "\n"
        except Exception as e:
            print(f"pdfplumber Error: {e}")
            
        return full_text

    def process_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
            
        print(f"Processing '{pdf_path}'...")
        raw_text = self.extract_hybrid_text(pdf_path)
        
        # Split into smaller chunks for the LLM
        chunks = self.text_splitter.split_text(raw_text)
        print(f"Process Complete! Total {len(chunks)} chunks created.")
        return chunks

# Test execution if the file is run directly
if __name__ == "__main__":
    processor = FinancialPDFProcessor()
    
    # Place a file named arcelik_2025_fr.pdf in the source_pdfs folder for testing
    test_pdf = "../../data/source_pdfs/arcelik_2025_fr.pdf" 
    
    # Adjust directories based on the project structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(current_dir, test_pdf)
    
    try:
        chunks = processor.process_pdf(target_path)
        print("\nSample Chunk Output:\n")
        print(chunks[0])  # Show only the first chunk
    except Exception as e:
        print(f"Test failed: {e}")
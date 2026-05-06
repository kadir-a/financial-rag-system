import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API keys from the .env file
load_dotenv()

class FinancialPDFProcessor:
    def __init__(self, chunk_size=5000, chunk_overlap=500):
        # Configure LlamaParse to output Markdown and align tables
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown", # Converts tables into aligned Markdown grids
            verbose=True,
            language="tr" # Prevents Turkish character corruption
        )
        
        # Updated separators to prevent Markdown tables ( | ) from splitting in half
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "|", " ", ""]
        )

    def process_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
            
        print(f"🚨 LlamaParse Active: Converting '{pdf_path}' into a matrix...")

        # Parse the PDF with LlamaParse (Runs via API)
        try:
            parsed_documents = self.parser.load_data(pdf_path)
        except Exception as e:
            raise RuntimeError(f"LlamaParse Parsing Error: {e}")
        
        # Combine all pages into a single aligned Markdown text
        full_markdown_text = "\n\n".join([doc.text for doc in parsed_documents])
        
        print("✅ Report successfully parsed. Table-compatible chunking initiating...")
        
        # Split the text into smaller chunks to feed the model
        chunks = self.text_splitter.split_text(full_markdown_text)
        print(f"✅ Process Complete! A total of {len(chunks)} matrix-compatible chunks created.")
        
        return chunks
 
# Test if the file is run directly (Legacy structure preserved)
if __name__ == "__main__":
    processor = FinancialPDFProcessor()
    
    # Path of the test file relative to the project directory
    test_pdf = "../../data/source_pdfs/arcelik_2025_fr.pdf" 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(current_dir, test_pdf)
    
    try:
        chunks = processor.process_pdf(target_path)
        print("\n--- Sample Chunk Output (Matrix Structure Check) ---\n")
        print(chunks[0])  # Show the first chunk
    except Exception as e:
        print(f"Test failed: {e}")
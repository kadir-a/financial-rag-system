import fitz  # PyMuPDF
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class FinancialPDFProcessor:
    def __init__(self, chunk_size=1200, chunk_overlap=250):
        # Finansal verilerde bağlam kopmasın diye overlap biraz yüksek tutulur
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def extract_hybrid_text(self, pdf_path: str) -> str:
        """Metinleri fitz ile, tabloları pdfplumber ile okur."""
        full_text = ""
        
        print("Metinler analiz ediliyor...")
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += f"\n--- SAYFA {page_num + 1} METNİ ---\n"
                full_text += page.get_text("text") + "\n"
        except Exception as e:
            print(f"PyMuPDF Hatası: {e}")

        
        print("Finansal tablolar taranıyor...")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    extracted_tables = page.extract_tables()
                    if extracted_tables: # Sadece tablo bulursa içeri gir
                        for i, table in enumerate(extracted_tables):
                            full_text += f"\n--- SAYFA {page_num + 1} TABLO {i + 1} ---\n"
                            for row in table:
                                clean_row = [str(cell).replace('\n', ' ') if cell else "-" for cell in row]
                                full_text += " | ".join(clean_row) + "\n"
        except Exception as e:
            print(f"pdfplumber Hatası: {e}")
        return full_text

    def process_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {pdf_path}")
            
        print(f"'{pdf_path}' işleniyor...")
        raw_text = self.extract_hybrid_text(pdf_path)
        
        # LLM'in yutabileceği küçük lokmalara (chunk) ayır
        chunks = self.text_splitter.split_text(raw_text)
        print(f"İşlem Tamam! Toplam {len(chunks)} adet chunk oluşturuldu.")
        return chunks

# Dosya doğrudan çalıştırılırsa test et
if __name__ == "__main__":
    processor = FinancialPDFProcessor()
    
    # Test için source_pdfs klasörüne ornek.pdf adında bir dosya atın
    test_pdf = "../../data/source_pdfs/arcelik_2025_fr.pdf" 
    
    # Dizinleri projeye göre ayarla
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(current_dir, test_pdf)
    
    try:
        chunks = processor.process_pdf(target_path)
        print("\nÖrnek Chunk Çıktısı:\n")
        print(chunks[0]) # Sadece ilk parçayı göster
    except Exception as e:
        print(f"Test başarısız: {e}")
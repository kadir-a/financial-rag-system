import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStoreManager:
    def __init__(self, db_path="./data/chroma_db"):
        self.db_path = db_path
        # Multilingual model for Turkish financial documents
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def create_and_store_embeddings(self, chunks: list, collection_name: str):
        """Converts text chunks into vectors and stores them in ChromaDB."""
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        # Persistent storage configuration
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path,
            collection_name=collection_name
        )
        print(f"Embeddings successfully stored in {self.db_path}")
        return vector_db

    def get_vector_db(self, collection_name: str):
        """Loads the existing vector database."""
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
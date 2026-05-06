import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorStoreManager:
    def __init__(self, base_db_path="./data/chroma_db"):
        self.base_db_path = base_db_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def _cleanup_old_vaults(self, current_collection):
        """Background garbage collector: Cleans up old vaults from the disk, silently skips locked ones."""
        if not os.path.exists(self.base_db_path):
            return
        
        for folder in os.listdir(self.base_db_path):
            folder_path = os.path.join(self.base_db_path, folder)
            # Do not delete the currently open vault, target other old vaults
            if os.path.isdir(folder_path) and folder != current_collection:
                try:
                    shutil.rmtree(folder_path)
                    print(f"🧹 Old ghost vault successfully destroyed: {folder}")
                except Exception:
                    # If Windows is holding the file (PermissionError), do not crash the system, pass silently.
                    # It will be deleted on the next server restart anyway.docker compose up -d --build
                    pass    
                    
    def create_and_store_embeddings(self, chunks: list, collection_name: str):
        """Converts text chunks into vectors and stores them in ChromaDB."""
        
        # 1. Create an isolated, brand new folder path for each upload (bypasses Windows locks)
        specific_db_path = os.path.join(self.base_db_path, collection_name)
        
        # 2. Run the background garbage collector and clean up old junk
        self._cleanup_old_vaults(current_collection=collection_name)
        
        print(f"🏗️ Building a brand new, isolated vault: {specific_db_path}")
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        # 3. Write data to the brand new folder
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=specific_db_path, # Dynamic, unlocked path
            collection_name=collection_name
        ) 
        print(f"✅ Embeddings successfully stored in {specific_db_path}")
        return vector_db

    def get_vector_db(self, collection_name: str):
        """Loads the existing vector database."""
        # Find the same isolated folder path when reading
        specific_db_path = os.path.join(self.base_db_path, collection_name)
        return Chroma(
            persist_directory=specific_db_path,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )



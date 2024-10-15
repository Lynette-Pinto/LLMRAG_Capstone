import json
from pathlib import Path
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

class EmbeddingManager:
    def __init__(self, model_name):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    def get_embeddings(self):
        return self.embeddings

class VectorSpaceManager:
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        self.embeddings = self.embedding_manager.get_embeddings()

    def create_vector_space(self, documents):
        vector_store = FAISS.from_documents(documents[:2], self.embeddings)

        with tqdm(total=len(documents), desc="Creating vector space") as pbar:
            batch_size = 100
            for i in range(2, len(documents), batch_size):
                batch_documents = documents[i:i + batch_size]
                temp_vector_store = FAISS.from_documents(batch_documents, self.embeddings)
                vector_store.merge_from(temp_vector_store)
                pbar.update(len(batch_documents))

        return vector_store

    def save_vector_space(self, vector_store, save_path):
        print(f"Saving vector space to {save_path}...")
        vector_store.save_local(save_path)
        print(f"Finished!")

    def load_vector_space(self, save_path):
        print(f"Loading vector space from {save_path}...")
        return FAISS.load_local(save_path, self.embeddings)


class DataLoader:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path

    def load_data(self):
        data = json.loads(Path(self.json_file_path).read_text())
        return data

    def create_documents(self, length=None):
        data = self.load_data()
        if length is None:
            length = len(data)
        
        documents = [
            Document(
                page_content=self.get_page_content(item),
                metadata=item
            )
            for item in data[:length]
        ]
        return documents

    def get_page_content(self, item):
        raise NotImplementedError("Subclasses must implement get_page_content method")
    

class MigrationDataLoader(DataLoader):
    def get_page_content(self, item):
        return f"Source: {item['source']['type']} | Destination: {item['destination']['type']} | Transformation: {item['transformation']} | Schedule: {item.get('schedule', 'Not specified')}"


def process_data(json_file_path, model_name, save_path, data_loader_class, length=None):
    embedding_manager = EmbeddingManager(model_name)

    # Initialize the vector space manager with the embedding manager
    vector_space_manager = VectorSpaceManager(embedding_manager)

    data_loader = data_loader_class(json_file_path)
    documents = data_loader.create_documents(length=length)

    vector_store = vector_space_manager.create_vector_space(documents)
    vector_space_manager.save_vector_space(vector_store, save_path)

    # Load the vector space and perform a search
    vector_store = vector_space_manager.load_vector_space(save_path)
    query = "MySQL to PostgreSQL migration"  # Example search query
    search_results = vector_store.search(query, k=2, search_type="similarity")
    print(search_results)

if __name__ == "__main__":
    
    json_file_path = '/content/JSONConversation.json'  
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  
    save_path = '/content/migration_vector_space/' 
    process_data(json_file_path, model_name, save_path, MigrationDataLoader, 100)

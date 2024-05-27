import chromadb
from sentence_transformers import SentenceTransformer
import os
import json
from src.models import ChunkSchema
from typing import List
from src.utils import load_json_to_chunkschema
import uuid

"""
I use this class if I want to modify the data I have stored in Elastic Search in some way.
"""

class ChromaDBLoader:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="chroma_db")
        print(self.client.heartbeat())
        self.collection = None
        try:
            self.collection = self.client.get_collection(name="twiga_documents")
        except ValueError as e:
            print("Collection doesn't exist, you have to create it.")
        
        print('Connected to Chroma!')


    def get_embedding(self, text):
        return self.model.encode(text)

    def create_index(self):
        try:
            self.client.delete_collection(name='twiga_documents')
        except ValueError as e:
            print("No index deletion required, it doesn't exist anyway.")
        
        # self.client.get_or_create_collection(index='twiga_documents')
        self.collection = self.client.create_collection(
            name="twiga_documents",
            metadata={"hnsw:space": "cosine"} # l2 is the default
        )
    
    def insert_documents(self, documents: List[ChunkSchema]):
        # todo: check this is correct
        chunks = [item.chunk for item in documents]
        embeddings = [item.embedding for item in documents]
        metadatas = [item.metadata.to_dict() for item in documents]
        new_metadatas = []
        for metadata in metadatas:
            for key, value in metadata.items():
                if value is None:
                    metadata[key] = ""
            new_metadatas.append(metadata)
        
        ids=[str(uuid.uuid4()) for _ in range(len(documents))]
        
        return self.collection.add(documents=chunks, embeddings=embeddings, metadatas=new_metadatas, ids=ids)
    
    def reindex(self, json_data: List[dict]):
        # This method is used if I want to regenerate the index 
        self.create_index()
        documents: List[ChunkSchema] = load_json_to_chunkschema(json_data)
        return self.insert_documents(documents)
    
    def search(self, query: str, n_results: int, where: dict) -> chromadb.QueryResult:
        embedding = self.get_embedding(query).tolist()
        return self.collection.query(query_embeddings=[embedding], n_results=n_results, where=where, include=["documents", "metadatas"])
    
    def retrieve_document(self, id: str):
        # This method gets the document associated with a specific ID in the elasticsearch database
        return self.collection.get(ids=[id])

if __name__ == "__main__":
    # # Paths to my data
    # exercise_path = os.path.join("data", "documents", "json", "tie-geography-f2-exercises-and-embeddings.json")
    # content_path = os.path.join("data", "documents", "json", "tie-geography-f2-content-and-embeddings.json")

    # with open(content_path, mode="r", encoding="utf-8") as f:
    #     content_data = json.loads(f.read())
    
    # with open(exercise_path, mode="r", encoding="utf-8") as f:
    #     exercise_data = json.loads(f.read())
    
    # content_documents: List[ChunkSchema] = load_json_to_chunkschema(content_data)
    # exercise_documents: List[ChunkSchema] = load_json_to_chunkschema(exercise_data)

    chromadb_loader = ChromaDBLoader()
    # chromadb_loader.create_index()
    # chromadb_loader.insert_documents(content_documents)
    # chromadb_loader.insert_documents(exercise_documents)

    # print(chromadb_loader.collection.peek()) # returns a list of the first 10 items in the collection
    # print(chromadb_loader.collection.count()) # returns the number of items in the collection

    # Search for documents
    res = chromadb_loader.search(query="Nomadic pastoralism", n_results=2, where={"doc_type": "Exercise"})
    print(res["documents"][0])
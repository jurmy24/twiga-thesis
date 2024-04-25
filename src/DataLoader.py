from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import json
from pprint import pprint
import time
from src.models import ChunkSchema
from datetime import datetime
from typing import List
from src.utils import load_json_file_to_chunkschema, save_objects_as_json

load_dotenv()
ELASTIC_SEARCH_API_KEY = os.getenv("ELASTIC_SEARCH_API_KEY")
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
DATA_DIR = os.getenv("DATA_DIR_PATH")

"""
I use this class if I want to modify the data I have stored in Elastic Search in some way.
"""

class DataLoader:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.es = Elasticsearch(cloud_id=ELASTIC_CLOUD_ID, api_key=ELASTIC_SEARCH_API_KEY)
        client_info = self.es.info()

        print('Connected to Elasticsearch!')
        pprint(client_info.body)

    def get_embedding(self, text):
        return self.model.encode(text)

    def create_index(self):
        self.es.indices.delete(index='twiga_documents', ignore_unavailable=True)
        self.es.indices.create(index='twiga_documents', mappings={
            'properties': {
                'embedding': {
                    'type': 'dense_vector',
                }
            }
        })
        
    def insert_document(self, document: ChunkSchema):
        embedding = self.get_embedding(document.chunk)
        document_data = json.loads(document.model_dump_json())
        return self.es.index(
            index='twiga_documents',
            document={
                **document_data,
                'embedding': embedding,
        })
    
    def insert_documents(self, documents: List[ChunkSchema]):
        operations = []
        for document in documents:
            embedding = self.get_embedding(document.chunk)
            document_data = json.loads(document.model_dump_json())
            operations.append({'index': {'_index': 'twiga_documents'}})
            operations.append({
                **document_data,
                'embedding': embedding,
            })

        return self.es.bulk(operations=operations)
    
    def reindex(self, data_file_path: str):
        # This method is used if I want to regenerate the index 
        self.create_index()
        documents: List[ChunkSchema] = load_json_file_to_chunkschema(data_file_path)
        return self.insert_documents(documents)
    
    def search(self, **query_args):
        return self.es.search(index="twiga_documents", **query_args)
    
    def retrieve_document(self, id: str):
        # This method gets the document associated with a specific ID in the elasticsearch database
        return self.es.get(index="twiga_documents", id=id)

if __name__ == "__main__":
    # Paths to my data
    exercise_path = os.path.join(DATA_DIR, "documents", "json", "tie-geography-f2-exercises.json")
    exercise_drop_path = os.path.join(DATA_DIR, "documents", "json", "tie-geography-f2-exercises-and-embeddings.json")
    content_path = os.path.join(DATA_DIR, "documents", "json", "v3-tie-geography-f2-content.json")
    content_drop_path = os.path.join(DATA_DIR, "documents", "json", "tie-geography-f2-content-and-embeddings.json")

    data_loader = DataLoader()

    # To add all my chunks to elastic search from scratch
    """
    data_loader.reindex(content_path)
    """
    
    # To add new data to the existing index
    """
    documents: List[ChunkSchema] = load_json_file_to_chunkschema(exercise_path)
    data_loader.insert_documents(documents)
    """

    
    documents: List[ChunkSchema] = load_json_file_to_chunkschema(content_path)
    new_docs = []
    for document in documents:
        embedding = data_loader.get_embedding(document.chunk)
        document_data = json.loads(document.model_dump_json())
        new_docs.append({
            **document_data,
            'embedding': embedding.tolist(),
        })
    
    save_objects_as_json(new_docs, content_drop_path)

    

    # This searches based on the Okapi BM25 algorithm (a higher score indicates a closer match to the query text)
    # results = dataLoader.search(
    #     query={
    #         'match': {
    #             'name': {
    #                 'query': "some text that I search on to match the name of a document"
    #             }
    #         }
    #     }
    # )
    
    # matches = results['hits']['hits']
    # total = results['hits']['total']['value']
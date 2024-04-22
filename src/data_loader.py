from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os
import json
from pprint import pprint
import time
from src.models import ChunkSchema
from datetime import datetime
from typing import List

load_dotenv()
ELASTIC_SEARCH_API_KEY = os.getenv("ELASTIC_SEARCH_API_KEY")
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
DATA_DIR = os.getenv("DATA_DIR_PATH")

class DataLoader:

    def __init__(self):
        self.es = Elasticsearch(cloud_id=ELASTIC_CLOUD_ID, api_key=ELASTIC_SEARCH_API_KEY)
        client_info = self.es.info()
        # self.model = TBD
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

    def get_embedding(self, text):
        return self.model.encode(text)

    def create_index(self):
        self.es.indices.delete(index='twiga_documents', ignore_unavailable=True)
        self.es.indices.create(index='twiga_documents')

    def insert_document(self, document: ChunkSchema):
        return self.es.index(
            index='twiga_documents', 
            body=document.model_dump_json()
        )
    
    def insert_documents(self, documents: List[ChunkSchema]):
        operations = []
        for document in documents:
            operations.append({'index': {'_index': 'twiga_documents'}})
            operations.append(document.model_dump_json())

        return self.es.bulk(operations=operations, pipeline="ent-search-generic-ingestion")
    
    def reindex(self):
        # This method is used if I want to regenerate the index 
        self.create_index()
        with open(os.path.join(DATA_DIR, "documents", "json", "PLACEHOLDER.json"), 'rt') as f:
            data = json.loads(f.read())
            # Convert JSON objects to DocumentSchema instances
            documents: List[ChunkSchema] = [ChunkSchema(**doc) for doc in data]
        return self.insert_documents(documents)
    
    def search(self, **query_args):
        return self.es.search(index="twiga_documents", **query_args)
    
    def retrieve_document(self, id: str):
        # This method gets the document associated with a specific ID in the elasticsearch database
        return self.es.get(index="twiga_documents", id=id)

if __name__ == "__main__":

    data_loader = DataLoader()
    data_loader.create_index()

    # document = ChunkSchema(
    #     title='Work From Home Policy',
    #     contents='The purpose of this full-time work-from-home policy is...',
    #     created_on=datetime.now().isoformat()
    # )

    # # response = dataLoader.insert_document(document)
    # # print(response['_id'])

    # # This searches based on the Okapi BM25 algorithm (a higher score indicates a closer match to the query text)
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
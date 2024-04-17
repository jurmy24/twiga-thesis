from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os
import json
from pprint import pprint
import time
from src.models import Document
from datetime import datetime

load_dotenv()
ELASTIC_SEARCH_API_KEY = os.getenv("ELASTIC_SEARCH_API_KEY")
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")

class DataLoader:

    def __init__(self):
        self.es = Elasticsearch(cloud_id=ELASTIC_CLOUD_ID, api_key=ELASTIC_SEARCH_API_KEY)
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

    def create_index(self):
        self.es.delete(index='twiga_documents', ignore_unavailable=True)
        self.es.create(index='twiga_documents')

    def insert_document(self, document: Document):
        return self.es.index(index='twiga_documents', body=document.model_dump_json())



if __name__ == "__main__":
    
    dataLoader = DataLoader()
    document = Document(
        title='Work From Home Policy',
        contents='The purpose of this full-time work-from-home policy is...',
        created_on=datetime.now().isoformat()
    )

    response = dataLoader.es.index(index='twiga_documents', body=document)
    print(response['_id'])
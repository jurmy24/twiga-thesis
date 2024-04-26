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
from src.utils import load_json_file_to_chunkschema

load_dotenv()
ELASTIC_SEARCH_API_KEY = os.getenv("ELASTIC_SEARCH_API_KEY")
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
DATA_DIR = os.getenv("DATA_DIR_PATH")

"""
I use this class if I want to search for data in Elastic Search
"""

class DataSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.es = Elasticsearch(cloud_id=ELASTIC_CLOUD_ID, api_key=ELASTIC_SEARCH_API_KEY)
        client_info = self.es.info()

        print('Connected to Elasticsearch!')
        pprint(client_info.body)

    def get_embedding(self, text):
        return self.model.encode(text)
    
    def retrieve_document(self, id: str):
        # This method gets the document associated with a specific ID in the elasticsearch database
        return self.es.get(index="twiga_documents", id=id)
    
    def search(self, query_args:dict):
        return self.es.search(index="twiga_documents", knn=query_args)

if __name__ == "__main__":
    # Paths to my data
    exercise_path = os.path.join(DATA_DIR, "documents", "json", "tie-geography-f2-exercises.json")
    content_path = os.path.join(DATA_DIR, "documents", "json", "v3-tie-geography-f2-content.json")
    

    # data_search = DataSearch()

    # query_args_knn = {
    #     'field': 'embedding',
    #     'query_vector': data_search.get_embedding("Generate a question that talks about rainfall in Tanzania."),
    #     'num_candidates': 50, # this is the number of candidate documents to consider from each shard
    #     'k': 10, # this is the number of results to return
    #     **filters
    # }
    
    # matches = results['hits']['hits'] # gives the resulting data (the 10 results)
    # total = results['hits']['total']['value'] # gives the number of results

    # print(matches)
    # print("::::::::::::::::::::::::::::::::::::::")
    # print(total)
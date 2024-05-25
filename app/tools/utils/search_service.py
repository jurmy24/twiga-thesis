import logging
import os
from pprint import pprint
from typing import Any, List

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions
from sentence_transformers import SentenceTransformer

# Set up basic logging configuration
logger = logging.getLogger(__name__)

load_dotenv()
ELASTIC_SEARCH_API_KEY = os.getenv("ELASTIC_SEARCH_API_KEY")
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
DATA_DIR = os.getenv("DATA_DIR_PATH")

"""
I use this class if I want to search for data in Elastic Search
"""


class DataSearch:

    def __init__(self):
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.es = Elasticsearch(
                cloud_id=ELASTIC_CLOUD_ID, api_key=ELASTIC_SEARCH_API_KEY
            )
            # client_info = self.es.info()
            logger.info("Connected to Elasticsearch!")
            # pprint(client_info.body)
        except es_exceptions.AuthenticationException as e:
            logger.error(f"Authentication failed: check your cloud ID and API key: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize DataSearch: {e}")

    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text)

    def retrieve_document(self, id: str):
        # This method gets the document associated with a specific ID in the elasticsearch database
        try:
            return self.es.get(index="twiga_documents", id=id)
        except es_exceptions.NotFoundError:
            logger.error(f"Document with ID {id} not found.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            return None

    async def search(
        self,
        size: int,
        knn_args: dict[str, Any] = None,
        query_args: dict[str, Any] = None,
        rank_args: dict[str, Any] = None,
    ) -> dict:
        try:
            if knn_args and query_args and rank_args:
                res = self.es.search(
                    index="twiga_documents",
                    size=size,
                    query=query_args,
                    knn=knn_args,
                    rank=rank_args,
                )
            elif knn_args:
                res = self.es.search(index="twiga_documents", size=size, knn=knn_args)
            elif query_args:
                res = self.es.search(
                    index="twiga_documents", size=size, query=query_args
                )
            else:
                raise Exception("Incorrect arguments sent to search method.")

            if res.body["timed_out"]:
                raise Exception("Search timed out.")

            return res.body
        except es_exceptions.RequestError as e:
            logger.error(f"Search request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            raise

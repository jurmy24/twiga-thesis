from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os


load_dotenv()
ELASTIC_SEARCH_API_KEY = os.getenv("ELASTIC_SEARCH_API_KEY")

client = Elasticsearch(
    "https://00898bfc0f1d4fe5a4721ec29a9b3a19.eu-north-1.aws.elastic-cloud.com:9243",
    api_key=ELASTIC_SEARCH_API_KEY  # Replace with your new API key
)

try:
    info = client.info()
    print(info)
except Exception as e:
    print("Failed to connect:", e)
from typing import List, Literal, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai.types.chat import ChatCompletion
import logging

from src.models import RetrievedDocSchema
from src.openai_requests import openai_request
from src.DataSearch import DataSearch
from src.utils import load_json_to_retrieveddocschema

"""
This is the modules file, which will contain the modular components that can be used by the RAG pipelines I build.
"""

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_search = DataSearch()

# TODO: Make it possible for the query_rewriter to see the entire conversation history so that it can add meat to the bone's of a message like "Write a question about what I said in my last message."
def query_rewriter(query:str) -> str:
    """
    Enhances a user query by rewriting it in better English and adding more detail.

    Parameters:
    - query (str): The user's original query string.
    """
    try:        

        messages = [
            {"role": "system", "content": "You are a helpful assistant that rewrites queries from the user into a short passage about the topic they are requesting a question about. You do not write a question, but find the topic they are requesting a question about and describe that topic."},
            {"role": "user", "content": f"{query}"}
        ]

        # Send the prompt to the OpenAI API using a suitable engine

        res: ChatCompletion = openai_request(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            max_tokens=100,  # Adjust based on the expected length of the enhanced query
            n=1,  # Number of completions to generate (default is 1)
            stop=None  # Optional stopping character or sequence
        )
            
        # Extract the enhanced query text from the response
        enhanced_query = res.choices[0].message.content

        return enhanced_query
    
    except Exception as e:
        logger.error(f"An error occurred when rewriting the query: {e}")
        return query  # Return the original query in case of an error
    
def local_query_rewriter(query:str) -> str:
    assert torch.cuda.is_available()
    torch.set_default_device("cuda")

    # Retrieve the microsoft phi-1.5 model (1.5 billion parameters)
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

    inputs = tokenizer(f"Write a passage that answers the given query: \nQuery: {query} \nAnswer: ", return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=100)
    text = tokenizer.batch_decode(outputs)[0]
    text = text.replace("Write a passage that answers the given query: \nQuery: {query} \nAnswer: ", "")
    return text.replace("\n", "") # because this model has a tendency to add unnecessary stuff if it tries to write another paragraph

def elasticsearch_retriever(retrieval_msg: str, size: int, doc_type: Literal["Content", "Exercise"], retrieve_dense: bool=True, retrieve_sparse: bool=False, metadata_filters: dict=None) -> Tuple:

    if metadata_filters is not None:
        filters = {"type": retrieval_type, **metadata_filters}
        filters = {"filter": {"term": {"metadata.type.keyword": doc_type}},
                   **metadata_filters}
    else:
        filters = {"filter": {"term": {"metadata.title.keyword": "Geography for Secondary Schools Student's Book Form Two"}}}

    rank_args = None
    knn_args = None
    query_args = None
    if retrieve_dense and retrieve_sparse:
        knn_args = {
            'field': 'embedding',
            'query_vector': data_search.get_embedding(retrieval_msg).tolist(),
            'num_candidates': 50, # this is the number of candidate documents to consider from each shard (I chose it at random)
            'k': size, # this is the number of results to return
            **filters
        }
        # This uses the BM25 algorithm find chunks that match the retrieval_msg
        query_args = {
            "match": {"chunk": retrieval_msg},
            **filters
        }
        rank_args = {
            "rrf": {}
        }
    elif retrieve_dense:
        knn_args = {
            'field': 'embedding',
            'query_vector': data_search.get_embedding(retrieval_msg).tolist(),
            'num_candidates': 50, # this is the number of candidate documents to consider from each shard (I chose it at random)
            'k': size, # this is the number of results to return
            **filters
        }
    elif retrieve_sparse:
        # I should use the BM25 algorithm here instead
        query_args = {
            "bool": {
                "must": {
                    "match": {"chunk": retrieval_msg}
                },
                **filters
            }
            
        }
    
    res = data_search.search(size=size, knn_args=knn_args, query_args=query_args, rank_args=rank_args)

    if res is None:
        return None, None, None, None

    num_hits: int = int(res["hits"]["total"]["value"])

    if num_hits == 0:
        return 0, None, None, None

    max_score: float = float(res["hits"]["max_score"])
    retrieval_type: str = str(res["hits"]["total"]["relation"])
    hits: List[dict] = res["hits"]["hits"]
    docs: List[RetrievedDocSchema] = load_json_to_retrieveddocschema(hits)

    return num_hits, max_score, retrieval_type, docs


def reranker(query, documents):
    pass
    

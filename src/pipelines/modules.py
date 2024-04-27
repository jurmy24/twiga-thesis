from typing import List, Literal, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai.types.chat import ChatCompletion
import logging

from src.llms.groq_requests import groq_request
from src.models import RetrievedDocSchema
from src.llms.openai_requests import openai_request
from src.DataSearch import DataSearch
from src.utils import load_json_to_retrieveddocschema, pretty_elasticsearch_response_rrf, pretty_elasticsearch_response
from src.prompt_templates import REWRITE_QUERY_PROMPT

"""
This is the modules file, which will contain the modular components that can be used by the RAG pipelines I build.
"""

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: Make it possible for the query_rewriter to see the entire conversation history so that it can add meat to the bone's of a message like "Write a question about what I said in my last message."
def query_rewriter(query:str, llm: Literal["openai", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]) -> str:
    """
    Enhances a user query by rewriting it in better English and adding more detail.

    Parameters:
    - query (str): The user's original query string.
    """
    try:        
        messages = [
            {"role": REWRITE_QUERY_PROMPT.role, "content": REWRITE_QUERY_PROMPT.content},
            {"role": "user", "content": f"Query: ({query})"}
        ]

        # Send the prompt to API using a suitable engine
        if llm == "openai":
            res: ChatCompletion = openai_request(
                model="gpt-3.5-turbo-0125",
                messages=messages,
                max_tokens=100,  # Adjust based on the expected length of the enhanced query
                n=1,  # Number of completions to generate (default is 1)
                stop=None  # Optional stopping character or sequence
            )
        else:
            res = groq_request(
                llm=llm,
                verbose=True,
                messages=messages, 
                max_tokens=100,
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

def elasticsearch_retriever(
        retrieval_msg: str, size: int, doc_type: Literal["Content", "Exercise"], 
        book_title: str="Geography for Secondary Schools Student's Book Form Two", 
        retrieve_dense: bool=True, retrieve_sparse: bool=False,
        verbose: bool=False) -> List[RetrievedDocSchema]:
    
    # TODO: move this elsewhere so that I don't need to reinitialize the datasearch class every time I call for a retrieval
    data_search = DataSearch()

    # Bool value stating whether or not we will use reciprocal rank fusion
    retrieve_rrf = retrieve_dense and retrieve_sparse

    # A filter that ensures we retrieve the correct type of doc's (either content or exercise) and from the correct book in the vector database
    filters = {"filter": [{"term": {"metadata.doc_type.keyword": doc_type}}, {"term": {"metadata.title.keyword": book_title}}]}
    rank_args = None
    knn_args = None
    query_args = None

    if retrieve_sparse:
        retrieval_method = "sparse"
        # This uses the BM25 algorithm to find chunks that match the retrieval_msg
        query_args = {
            "bool": {
                "must" : [
                    {"match": {
                        "chunk" : {  
                            "query": retrieval_msg,
                            "analyzer": "standard"
                        }
                    }}
                ],
                **filters
            }
        }
    
    if retrieve_dense:
        retrieval_method = "dense"
        # This uses cosine similarity on vector embeddings of the chunks
        knn_args = {
            'field': 'embedding',
            'query_vector': data_search.get_embedding(retrieval_msg).tolist(),
            'num_candidates': 500, # this is the number of candidate documents to consider from each shard (I chose it at random)
            'k': size, # this is the number of results to return
            **filters
        }
    
    if retrieve_rrf:
        retrieval_method = "hybrid"
        rank_args = {
            "rrf": {}
        }        
    
    # Call DataSearch to search ElasticSearch
    try:
        res = data_search.search(size=size, knn_args=knn_args, query_args=query_args, rank_args=rank_args)
    except Exception as e:
        raise

    # Check if there were any search results, if not, return an empty list
    num_hits: int = int(res["hits"]["total"]["value"])
    if num_hits == 0:
        return []
    
    docs: List[RetrievedDocSchema] = load_json_to_retrieveddocschema(res["hits"]["hits"], retrieval_method)

     # This prints out the response in a pretty format if the caller desires
    if verbose:
        pretty_elasticsearch_response_rrf(res) if retrieve_rrf else pretty_elasticsearch_response(res)

    return docs


def reranker(query, documents: List[RetrievedDocSchema]) -> List[RetrievedDocSchema]:
    # this function takes in a list of documents, puts them into a reranker against the original query and reorders them
    # it returns a list of documents that is shorter
    pass
    

import os
import json
from typing import Any, List, Literal
from pydantic import ValidationError
import logging

from sentence_transformers import SentenceTransformer
from llama_index.core.schema import Document
import tiktoken

from src.prompt_templates import DEFAULT_TEXT_QA_PROMPT
from src.models import ChunkSchema, EvalQuery, Metadata, PipelineData, ResponseSchema, RetrievedDocSchema, RewrittenQuery

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_markdown_content(document: Document, output_path: str):
    """This saves a LlamaIndex Document as markdown"""
    markdown_content = document.text
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(markdown_content)

def save_objects_as_json(objects: List[object], filename: str, rewrite: bool=True):
    """This saves any list of objects in a json file"""
    # Check if obj has the attribute to_dict before converting, else leave it as is
    data_to_save = [obj.to_dict() if hasattr(obj, 'to_dict') else obj for obj in objects]
    
    # If rewrite is set to false and the file exists we should append info
    if not rewrite and os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
                if isinstance(existing_data, list):
                    existing_data.extend(data_to_save)
                    data_to_save = existing_data
                else:
                    raise ValueError("JSON root is not a list; cannot append data.")
            except json.JSONDecodeError:
                logger.error("Empty or invalid JSON file, starting fresh...")

    # Write the updated data back to the file
    with open(filename, 'w', encoding='utf-8') as file:
        try:
            json.dump(data_to_save, file, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Error when dumping data to json: {e}")

def get_embedding(text: str, model: SentenceTransformer) -> List[float]:
    """This uses a sentence transformer to create a text embedding"""
    return model.encode(text).tolist()

def num_tokens_from_string(string: str, encoding_name: str="cl100k_base") -> int:
    """This returns the number of OpenAI-equivalent tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages: List[dict], encoding_name: str="cl100k_base") -> int:
    """Return the number of tokens used by a list of messages in the format sent to the OpenAI or Groq API."""
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += num_tokens_from_string(value, encoding_name)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def convert_to_evalquery(item: dict) -> EvalQuery:
    """This converts a single dict item into an EvalQuery item

    Assuming the item format:
    {
        "query": str,
        "requested_exercise_format": str,
        "topic": str,
        "embedding": List[float],
        "rewritten_query": {
            "rewritten_query_str": self.rewritten_query_str,
            "embedding": self.embedding
        }
    }
    """
    try:
        if item.get('rewritten_query', None) is not None:
            return EvalQuery(**{**item, "rewritten_query": RewrittenQuery(**item['rewritten_query'])})
        else:
            return EvalQuery(**item)
    except ValidationError as e:
        logger.error(f"ValidationError occurred: {e}")
        raise ValidationError

def load_json_to_evalquery(data: List[dict]) -> List[EvalQuery]:
    """This converts a bunch of data in json format (dict) to EvalQuery's"""
    eval_queries = []
    for index, item in enumerate(data):
        try:
            eval_queries.append(convert_to_evalquery(item))
        except ValidationError as e:
            logger.error(f"ValidationError when parsing evaluation query {index + 1}: {e}")
            raise ValidationError
    return eval_queries

def convert_to_chunkschema(item: dict) -> ChunkSchema:
    """This converts a single dict item into a ChunkSchema item

    Assuming the item format:
    {
        "chunk": str,
        "metadata": {
            "title": str,
            "chapter": str,
            "subsection": stre,
            "subsubsection": str,
            "doc_type": str,
            "exercise_format": str
        },
        "embedding": List[float]
    }
    """
    try:
        return ChunkSchema(**{**item, "metadata": Metadata(**item['metadata'])})
    except ValidationError as e:
        logger.error(f"ValidationError occurred: {e}")
        raise ValidationError

def load_json_to_chunkschema(data: List[dict]) -> List[ChunkSchema]:
    """This converts a bunch of data in json format (dict) to a ChunkSchema's"""
    chunks = []
    for index, item in enumerate(data):
        try:
            chunks.append(convert_to_chunkschema(item))
        except ValidationError as e:
            logger.error(f"ValidationError when parsing chunk {index + 1}: {e}")
            raise ValidationError
    return chunks

def convert_to_retrieveddocschema(item: dict, retrieval_method: Literal["sparse", "dense", "hybrid"]=None) -> RetrievedDocSchema:
    """This converts a single dict item into a RetrievedDocSchema item

    Assuming the item format:
    {
        "retrieval_type": str,
        "score": float,
        "rank": int,
        "id": str,
        "source": ChunkSchema
    }
    """

    # Have to deal with some spelling stuff due to the mismatch between the values returned by Elastic Search and the titles we use
    source_spelling: str = 'source' if item.get('_source', None) is None else '_source'
    score_spelling: str = 'score' if item.get('_score', None) is None else '_score'
    rank_spelling: str = 'rank' if item.get('_rank', None) is None else '_rank'
    id_spelling: str = 'id' if item.get('_id', None) is None else '_id'
    try:
        chunk: ChunkSchema = convert_to_chunkschema(item[source_spelling])
        current_retrieval_type: str | None = item.get('retrieval_type', None)
        retrieval_method = current_retrieval_type if current_retrieval_type else retrieval_method

        return RetrievedDocSchema(retrieval_type=retrieval_method, rank=item.get(rank_spelling, None), score=item.get(score_spelling, None), id=item.get(id_spelling, None), source=chunk)
    except ValidationError as e:
        logger.error(f"ValidationError occurred: {e}")
        raise ValidationError
    
def load_json_to_retrieveddocschema(data: List[dict], retrieval_method: Literal["sparse", "dense", "hybrid"]=None) -> List[RetrievedDocSchema]:
    """This converts a bunch of data in json format (dict) to RetrievedDocSchema's"""
    docs = []
    for index, item in enumerate(data):
        try:
            docs.append(convert_to_retrieveddocschema(item, retrieval_method))
        except ValidationError as e:
            logger.error(f"ValidationError when parsing retrieved document {index + 1}: {e}")
            raise ValidationError
    return docs

def convert_to_pipelinedata(item: dict, retrieval_method: Literal["sparse", "dense", "hybrid"]=None) -> PipelineData:
    """This converts a single dict item into a PipelineData item

    Assuming the item format:
    {
        "query": EvalQuery,
        "retrieved": RetrievedDocSchema,
        "response": ResponseSchema,
    }
    """
    try:
        query = convert_to_evalquery(item.get('query')) # query is mandatory for PipelineData
        retrieved_docs = item.get('retrieved', None)
        response = item.get('response', None)
        
        if retrieved_docs:
            retrieved_docs = load_json_to_retrieveddocschema(retrieved_docs, retrieval_method)
        if response:
            response = ResponseSchema(**response)

        return PipelineData(query=query, retrieved_docs=retrieved_docs, response=response)
    except ValidationError as e:
        logger.error(f"ValidationError occurred: {e}")
        raise ValidationError

def load_json_to_pipelinedata(data: List[dict], retrieval_method: Literal["sparse", "dense", "hybrid"]=None) -> List[PipelineData]:
    pipe_data = []
    for index, item in enumerate(data):
        try:
            pipe_data.append(convert_to_pipelinedata(item))
        except ValidationError as e:
            logger.error(f"ValidationError when parsing pipeline item {index + 1}: {e}")
    return pipe_data

def pretty_elasticsearch_response_rrf(response):
    if len(response["hits"]["hits"]) == 0:
        print("Your search returned no results.")
    else:
        for hit in response["hits"]["hits"]:
            id = hit["_id"]
            rank = hit["_rank"]
            title = hit["_source"]["metadata"]["title"]
            chapter = hit["_source"]["metadata"]["chapter"]
            subsection = hit["_source"]["metadata"]["subsection"]
            chunk = hit["_source"]["chunk"]
            pretty_output = f"\nID: {id}\nTitle: {title}\nChapter: {chapter}\nSubsection: {subsection}\nchunk: {chunk}\nRank: {rank}"
            print(pretty_output)

def pretty_elasticsearch_response(response):
    if len(response["hits"]["hits"]) == 0:
        print("Your search returned no results.")
    else:
        for hit in response["hits"]["hits"]:
            id = hit["_id"]
            score = hit["_score"]
            title = hit["_source"]["metadata"]["title"]
            chapter = hit["_source"]["metadata"]["chapter"]
            subsection = hit["_source"]["metadata"]["subsection"]
            chunk = hit["_source"]["chunk"]
            pretty_output = f"\nID: {id}\nTitle: {title}\nChapter: {chapter}\nSubsection: {subsection}\nchunk: {chunk}\nScore: {score}"
            print(pretty_output)

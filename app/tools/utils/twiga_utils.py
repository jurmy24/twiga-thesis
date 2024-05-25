import logging
from typing import List, Literal

import tiktoken
from pydantic import ValidationError

from app.tools.utils.models import ChunkSchema, Metadata, RetrievedDocSchema

logger = logging.getLogger(__name__)


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """This returns the number of OpenAI-equivalent tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_messages(
    messages: List[dict], encoding_name: str = "cl100k_base"
) -> int:
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


def _convert_to_chunkschema(item: dict) -> ChunkSchema:
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
        return ChunkSchema(**{**item, "metadata": Metadata(**item["metadata"])})
    except ValidationError as e:
        logger.error(f"ValidationError occurred: {e}")
        raise ValidationError


def _convert_to_retrieveddocschema(
    item: dict, retrieval_method: Literal["sparse", "dense", "hybrid"] = None
) -> RetrievedDocSchema:
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
    source_spelling: str = "source" if item.get("_source", None) is None else "_source"
    score_spelling: str = "score" if item.get("_score", None) is None else "_score"
    rank_spelling: str = "rank" if item.get("_rank", None) is None else "_rank"
    id_spelling: str = "id" if item.get("_id", None) is None else "_id"
    try:
        chunk: ChunkSchema = _convert_to_chunkschema(item[source_spelling])
        current_retrieval_type: str | None = item.get("retrieval_type", None)
        retrieval_method = (
            current_retrieval_type if current_retrieval_type else retrieval_method
        )

        return RetrievedDocSchema(
            retrieval_type=retrieval_method,
            rank=item.get(rank_spelling, None),
            score=item.get(score_spelling, None),
            id=item.get(id_spelling, None),
            source=chunk,
        )
    except ValidationError as e:
        logger.error(f"ValidationError occurred: {e}")
        raise ValidationError


def load_json_to_retrieveddocschema(
    data: List[dict], retrieval_method: Literal["sparse", "dense", "hybrid"] = None
) -> List[RetrievedDocSchema]:
    """This converts a bunch of data in json format (dict) to RetrievedDocSchema's"""
    docs = []
    for index, item in enumerate(data):
        try:
            docs.append(_convert_to_retrieveddocschema(item, retrieval_method))
        except ValidationError as e:
            logger.error(
                f"ValidationError when parsing retrieved document {index + 1}: {e}"
            )
            raise ValidationError
    return docs


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

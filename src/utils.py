import json
from typing import List
from llama_index.core.schema import Document
import os

from pydantic import ValidationError
from src.prompt_templates import DEFAULT_TEXT_QA_PROMPT, DEFAULT_KG_TRIPLET_EXTRACT_PROMPT, CHAT_TEXT_QA_SYSTEM_PROMPT, CHAT_TEXT_QA_USER_PROMPT
from src.models import ChatMessage, ChunkSchema, Metadata, RetrievedDocSchema
import tiktoken

def save_documents_as_json(documents: List[Document], output_path: str):
    """
    Saves a list of Document objects as a JSON file.

    Parameters:
    - documents (List[Document]): The documents to save.
    - output_path (str): The file path to save the JSON data to.
    """
    # Convert documents into a list of dictionaries
    docs_list = []
    for doc in documents:
        doc_dict = {
            "id": doc.id_,
            "text": doc.text,
            "metadata": doc.metadata
            # Include other relevant fields here
        }
        docs_list.append(doc_dict)

    # Check if the file already exists
    if os.path.exists(output_path):
        # Read the existing content
        with open(output_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                # Ensure the data read is a list to append to
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                # If the file is empty or contains only whitespaces, start a new list
                data = []
    else:
        # Start a new list if the file doesn't exist
        data = []
    
    # Append the new data
    data.append(doc_dict)
    
    # Write the updated data back to the file
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def save_markdown_content(document: Document, output_path):
    """
    Saves a Document object text as a markdown file

    Parameters:
    - document (Document): The document to save.
    - output_path (str): The file path to save the JSON data to.
    """

    markdown_content = document.text
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(markdown_content)

def save_nodes_as_json(nodes, filename="base_nodes.json"):
    """
    Saves base_nodes to a JSON file.

    Parameters:
    - base_nodes: A list of base node objects. Assumes each object can be directly serialized.
    - filename (str): The filename for the output JSON.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        # Convert base_nodes to a serializable format if necessary
        base_nodes_data = [node.__dict__ for node in nodes]
        json.dump(base_nodes_data, file, ensure_ascii=False, indent=4)

def save_base_nodes_as_json(base_nodes, filename="base_nodes.json"):
    """
    Saves base_nodes to a JSON file.

    Parameters:
    - base_nodes: A list of base node objects. Assumes each object can be directly serialized.
    - filename (str): The filename for the output JSON.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        # Convert base_nodes to a serializable format if necessary
        base_nodes_data = [node.__dict__ for node in base_nodes]
        json.dump(base_nodes_data, file, ensure_ascii=False, indent=4)

# def save_objects_as_json(objects, filename="objects.json"):
#     """
#     Saves objects to a JSON file.

#     Parameters:
#     - objects: A list of object nodes or similar structures. Assumes each object can be directly serialized.
#     - filename (str): The filename for the output JSON.
#     """
#     with open(filename, 'w', encoding='utf-8') as file:
#         # Convert objects to a serializable format if necessary
#         objects_data = [obj.to_dict() if hasattr(obj, 'to_dict') else obj for obj in objects]
#         json.dump(objects_data, file, ensure_ascii=False, indent=4)

def save_objects_as_json(objects, filename, rewrite=True):
    """
    Appends objects to a JSON file without overwriting existing data.

    Parameters:
    - objects: A list of object nodes or similar structures. Assumes each object can be directly serialized or has a to_dict() method.
    - filename (str): The filename for the output JSON.
    """
    data_to_save = [obj.to_dict() if hasattr(obj, 'to_dict') else obj for obj in objects]
    
    if not rewrite and os.path.exists(filename):
        # File exists, read the existing data
        with open(filename, 'r', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
                if isinstance(existing_data, list):  # Assumes the root of the JSON is a list
                    existing_data.extend(data_to_save)  # Add new data to the existing list
                    data_to_save = existing_data
                else:
                    raise ValueError("JSON root is not a list; cannot append data.")
            except json.JSONDecodeError:
                print("Empty or invalid JSON file, starting fresh...")
                data_to_save = [data_to_save]  # Start a new list if file is empty or not valid JSON

    # Write the updated data back to the file
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data_to_save, file, ensure_ascii=False, indent=4)

def generate_text_qa_prompt(context: str, query: str) -> str:
    prompt = DEFAULT_TEXT_QA_PROMPT.format(context_str=context, query_str=query)
    return prompt

def generate_kg_triplet_prompt(text: str, max_triplets:int=3) -> str:
    prompt = DEFAULT_KG_TRIPLET_EXTRACT_PROMPT.format(text=text, max_knowledge_triplets=max_triplets)
    return prompt

def generate_chat_text_qa_user_prompt(context: str, query: str) -> ChatMessage:
    prompt = CHAT_TEXT_QA_USER_PROMPT.content.format(context_str=context, query_str=query)
    return ChatMessage(content=prompt, role="user")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_json_file_to_chunkschema(file_path: str) -> List[ChunkSchema]:
    with open(file_path, 'r') as file:
        data = json.load(file)
        chunks = []
        for index, item in enumerate(data):
            try:
                # Explicitly handle the creation of Metadata objects if necessary
                chunks.append(ChunkSchema(**{**item, "metadata": Metadata(**item['metadata'])}))
            except ValidationError as e:
                print(f"ValidationError when parsing document {index + 1}: {e}")
        # chunks = [ChunkSchema(**{**item, "metadata": Metadata(**item['metadata'])}) for item in data]
        return chunks
    
def load_json_to_retrieveddocschema(data: List[dict]) -> List[RetrievedDocSchema]:
    docs = []
    for index, item in enumerate(data):
        try:
            # Explicitly handle the creation of Metadata objects if necessary
            chunk = ChunkSchema(**{**item['_source'], "metadata": Metadata(**item['_source']['metadata'])})
            doc = RetrievedDocSchema(retrieval_type='IDK', score=item['_score'], id=item['_id'], source=chunk)
            docs.append(doc)
        except ValidationError as e:
            print(f"ValidationError when parsing retrieved document {index + 1}: {e}")
    # chunks = [ChunkSchema(**{**item, "metadata": Metadata(**item['metadata'])}) for item in data]
    return docs

def pretty_elasticsearch_response(response):
    if len(response["hits"]["hits"]) == 0:
        print("Your search returned no results.")
    else:
        for hit in response["hits"]["hits"]:
            id = hit["_id"]
            publication_date = hit["_source"]["publish_date"]
            rank = hit["_rank"]
            title = hit["_source"]["title"]
            summary = hit["_source"]["summary"]
            pretty_output = f"\nID: {id}\nPublication date: {publication_date}\nTitle: {title}\nSummary: {summary}\nRank: {rank}"
            print(pretty_output)

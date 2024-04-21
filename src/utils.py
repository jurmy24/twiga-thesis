import json
from typing import List
from llama_index.core.schema import Document
import os

from pydantic import ValidationError
from src.prompt_templates import DEFAULT_TEXT_QA_PROMPT, DEFAULT_KG_TRIPLET_EXTRACT_PROMPT, CHAT_TEXT_QA_SYSTEM_PROMPT, CHAT_TEXT_QA_USER_PROMPT
from src.models import ChatMessage, ChunkSchema, Metadata
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

def save_objects_as_json(objects, filename="objects.json"):
    """
    Saves objects to a JSON file.

    Parameters:
    - objects: A list of object nodes or similar structures. Assumes each object can be directly serialized.
    - filename (str): The filename for the output JSON.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        # Convert objects to a serializable format if necessary
        objects_data = [obj.__dict__ for obj in objects]
        json.dump(objects_data, file, ensure_ascii=False, indent=4)

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
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            # Explicitly handle the creation of Metadata objects if necessary
            chunks = [ChunkSchema(**{**item, "metadata": Metadata(**item['metadata'])}) for item in data]
            return chunks
    except ValidationError as e:
        print("Validation error when parsing JSON data:", e)

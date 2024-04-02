import json
from typing import List
from llama_index.core.schema import Document
import os
from src.prompt_templates import DEFAULT_TEXT_QA_PROMPT, DEFAULT_KG_TRIPLET_EXTRACT_PROMPT, CHAT_TEXT_QA_PROMPT, CHAT_REFINE_PROMPT
from llama_index.core.base.llms.types import ChatMessage, MessageRole

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
    # Can be called in the following way:
    # context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
    # query = "What is the Eiffel Tower made of?"

    # text_qa_prompt = generate_text_qa_prompt(context, query)

    prompt = DEFAULT_TEXT_QA_PROMPT.template.format(context_str=context, query_str=query)
    return prompt

def generate_kg_triplet_prompt(text: str, max_triplets:int=3) -> str:
    prompt = DEFAULT_KG_TRIPLET_EXTRACT_PROMPT.template.format(text=text, max_knowledge_triplets=max_triplets)
    return prompt

def generate_chat_text_qa_prompt(context: str, query: str) -> List[str]:
    # Prepare the context and query message
    context_query_msg = ChatMessage(
        content=f"Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query}\nAnswer: ",
        role=MessageRole.USER,
    )

    # Combine system prompt with the user's context and query
    prompt_messages = [CHAT_TEXT_QA_PROMPT.message_templates[0], context_query_msg]
    return prompt_messages


def generate_chat_refine_prompt(context: str, query: str, existing_answer: str) -> List[str]:
    # Create a new context message with the task of refining an existing answer
    refine_msg = ChatMessage(
        content=f"New Context: {context}\nQuery: {query}\nOriginal Answer: {existing_answer}\nNew Answer: ",
        role=MessageRole.USER,
    )

    # The first message is a generic instruction for refinement tasks
    prompt_messages = [CHAT_REFINE_PROMPT.message_templates[0], refine_msg]
    return prompt_messages
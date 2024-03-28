import json
from typing import List
from llama_index.core.schema import Document
import os


def save_documents_as_json(document: Document, output_path):
    """
    Saves a Document object as a JSON file.

    Parameters:
    - document (Document): The document to save.
    - output_path (str): The file path to save the JSON data to.
    """

    # Assuming the Document object has .text and other attributes you're interested in
    doc_dict = {
        "id": document.id_,
        "text": document.text,
        "metadata": document.metadata
        # Include other relevant fields here
    }

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
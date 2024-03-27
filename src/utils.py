import json
import csv
from typing import List
from llama_index.core.schema import Document


def save_documents_as_json(documents: List[Document], output_path):
    """
    Saves a list of Document objects as a JSON file.

    Parameters:
    - documents (List[Document]): The documents to save.
    - output_path (str): The file path to save the JSON data to.
    """
    # Convert documents into a list of dictionaries
    docs_list = []
    for doc in documents:
        # Assuming each Document object has .text and other attributes you're interested in
        doc_dict = {
            "id": doc.id_,
            "text": doc.text,
            "metadata": doc.metadata
            # Include other relevant fields here
        }
        docs_list.append(doc_dict)

    # Write the list of document dictionaries to a JSON file
    with open(output_path, "w") as json_file:
        json.dump(docs_list, json_file, indent=4)

def save_documents_as_csv(documents: List[Document], output_path):
    """
    Saves a list of Document objects as a CSV file.

    Parameters:
    - documents (List[Document]): The documents to save.
    - output_path (str): The file path to save the CSV data to.
    """
    # Define the CSV headers based on the Document object attributes you're interested in
    headers = ["id", "text", "metadata"]  # Add other headers as needed

    # Write the documents to a CSV file
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()

        for doc in documents:
            # Convert each Document into a dictionary matching the headers
            row = {
                "id": doc.id_,
                "text": doc.text,
                "metadata": str(doc.metadata)  # Assuming metadata is a dictionary
                # Include other fields here as needed
            }
            writer.writerow(row)


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
import json
import csv

def save_documents_as_json(documents, output_path):
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

def save_documents_as_csv(documents, output_path):
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
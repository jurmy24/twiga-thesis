import os
from dotenv import load_dotenv
import pypandoc

from src.KnowledgeParser import KnowledgeParser
from src.utils import save_documents_as_json, save_markdown_content

load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY") 
DATA_DIR = os.getenv('DATA_DIR_PATH')

"""
This file is for running a few scripts and testing things.
"""

# Can view the markdown code with https://dillinger.io/

if __name__ == "__main__":

    # Initialize the KnowledgeParser
    knowledge_parser = KnowledgeParser(api_key=LLAMA_CLOUD_API_KEY)


    pdf = os.path.join(DATA_DIR, "airbnb-faq.pdf")
    document = knowledge_parser.parse_pdf_sync(pdf)

    save_documents_as_json(document, os.path.join(DATA_DIR, "documents", "markdown-entire-docs.json"))

    save_markdown_content(document, os.path.join(DATA_DIR, "documents", "airbnb-faq.md")) # TODO: make this also accept a longer list of items 

    md_content_path = os.path.join(DATA_DIR, "documents", "airbnb-faq.md")
    recreated_content_path = os.path.join(DATA_DIR, "documents", "airbnb-faq-remade.pdf")
    
    nodes = knowledge_parser.node_parser.get_nodes_from_documents(list(document))
    base_nodes, objects = knowledge_parser.node_parser.get_nodes_and_objects(nodes)
    # print(base_nodes[0])
    # print(objects[0])

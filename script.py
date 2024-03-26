import os
from dotenv import load_dotenv

from src.KnowledgeParser import KnowledgeParser
from src.utils import save_documents_as_csv, save_documents_as_json
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI


load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# Base directory where the output PDFs will be saved
DATA_DIR = os.getenv('DATA_DIR_PATH')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

"""
This file is for running a few scripts and testing things.
"""

if __name__ == "__main__":

    # Initialize the KnowledgeParser
    knowledge_parser = KnowledgeParser(api_key=LLAMA_CLOUD_API_KEY)

    # Extract the relevant pages
    new_pdf_path = knowledge_parser.extract_pages(pdf_path=f"{DATA_DIR}airbnb-faq.pdf", start_page=0, end_page=None)
    documents = knowledge_parser.parse_pdf_sync(new_pdf_path)
    node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-3.5-turbo-0125"), num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    print(len(base_nodes))
    print(len(objects))
    print("-----------------------------")
    print(base_nodes[0:5])
    print(objects[0:5])
    # TODO: give the OpenAI api key to LLamaindex, idk how yet


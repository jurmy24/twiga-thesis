import os
from dotenv import load_dotenv

from src.KnowledgeParser import KnowledgeParser

load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# Base directory where the output PDFs will be saved
DATA_DIR = os.getenv('DATA_DIR_PATH')

"""
This file is for running a few scripts and testing things.
"""

if __name__ == "__main__":

    # Initialize the KnowledgeParser
    knowledge_parser = KnowledgeParser(api_key=LLAMA_CLOUD_API_KEY)

    # Extract the relevant pages
    new_pdf_path = knowledge_parser.extract_pages(pdf_path=f"{DATA_DIR}airbnb-faq.pdf", start_page=0, end_page=None)
    # documents = knowledge_parser.parse_pdf_sync(new_pdf_path)
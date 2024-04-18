import json
from typing import List
from langchain.text_splitter import MarkdownHeaderTextSplitter
import os
from dotenv import load_dotenv

# Start by chunking based on headers
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR_PATH")

try:
    with open(os.path.join(DATA_DIR, "documents", "markdown", "tie-geography-f2-content.md"), 'rt') as f:
        markdown_text = f.read()
except FileNotFoundError:
    print("The file does not exist.")
except IOError:
    print("An error occurred while reading the file.")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, 
    strip_headers=False
)

md_header_splits = markdown_splitter.split_text(markdown_text)


# Convert each document to dictionary
docs_to_save = [doc.dict() for doc in md_header_splits]

# Serialize to JSON and write to file
with open(os.path.join(DATA_DIR, "documents", "json", "tie-geography-f2-content.json"), 'w') as f:
    json.dump(docs_to_save, f, indent=4)

"""
If we want to further split the text
"""
# # Char-level splits
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# chunk_size = 250
# chunk_overlap = 30
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=chunk_size, chunk_overlap=chunk_overlap
# )

# # Split
# splits = text_splitter.split_documents(md_header_splits)
# splits
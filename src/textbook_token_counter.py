import json
from collections import defaultdict
import os
from typing import List
from dotenv import load_dotenv
import tiktoken
from src.models import ChunkSchema
from src.utils import num_tokens_from_string, load_json_file_to_chunkschema

# Start by chunking based on headers
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR_PATH")

def get_chapter_token_count(data: List[ChunkSchema]) -> float:
    # Dictionary to hold concatenated contents by chapter
    content_by_chapter = defaultdict(str)

    # Iterate through each document in the data and put it into the associated chapter
    for entry in data:
        metadata = entry.metadata
        chapter = metadata.chapter
        
        # Only process entries that have a 'chapter'
        if chapter:
            content_by_chapter[chapter] += entry.page_content + " "  # Add a space to separate contents
    
    # Now convert this to a list of contents
    chapter_list = list(content_by_chapter.values())

    return calculate_average_tokens(chapter_list)

def get_subsection_token_count(data: List[ChunkSchema]) -> float:
    subsection_list = []

    # Iterate through each document in the data and put it into the associated chapter
    for entry in data:
        subsection_list.append(entry.page_content)

    return calculate_average_tokens(subsection_list)

def calculate_average_tokens(contents: List) -> float:
    token_count = 0.0

    failed = 0
    for c in contents:
        try:
            token_count += num_tokens_from_string(c, encoding_name="cl100k_base")
        except:
            failed += 1
            pass
    
    print(f"This many chunks were not included: {failed} / {len(contents)}")
    
    return token_count / (float(len(contents)-float(failed)))

def main():
    # This is the path to the json file of the subsection-chunked textbook
    file_path = os.path.join(DATA_DIR, "documents", "json", "tie-geography-f2-content.json")

    # Load the JSON data from the file
    chunks = load_json_file_to_chunkschema(file_path)

    avg_chapter_tokens = get_chapter_token_count(chunks)
    avg_subsection_tokens = get_subsection_token_count(chunks)
    
    # Display the average token counts per chapter
    print(f"Average chapter tokens: {avg_chapter_tokens}")
    print(f"Average subsection tokens: {avg_subsection_tokens}")
    
if __name__ == "__main__":
    main()

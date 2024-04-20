import json
from collections import defaultdict
import os
from typing import List
from dotenv import load_dotenv
import tiktoken
from src.utils import get_token_count

# Start by chunking based on headers
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR_PATH")

def load_and_merge_contents(file_path):
    # Dictionary to hold concatenated contents by chapter
    content_by_chapter = defaultdict(str)
    
    # Load the JSON data from the file
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            # Iterate through each document in the data
            for entry in data:
                metadata = entry.get("metadata", {})
                chapter = metadata.get("chapter")
                
                # Only process entries that have a 'chapter'
                if chapter:
                    content_by_chapter[chapter] += entry["page_content"] + " "  # Add a space to separate contents

        return content_by_chapter
    except FileNotFoundError:
        print("The file does not exist.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def calculate_average_tokens(content: List[dict]) -> float:
    token_count = 0.0

    list_content = [document['page_content'] for document in content]

    failed = 0
    for c in list_content:

        try:
            token_count += get_token_count(content, encoding_name="cl100k_base")
        except:
            failed += 1
            pass
    print(failed)
    
    return token_count / (float(len(c)-float(failed)))

def main():
    file_path = os.path.join(DATA_DIR, "documents", "json", "tie-geography-f2-content.json")  # Replace with the path to your JSON file
    # merged_contents = load_and_merge_contents(file_path)

    # # Serialize to JSON and write to file
    # with open(os.path.join(DATA_DIR, "documents", "json", "chaptered-tie-geography-f2-content.json"), 'w') as f:
    #     json.dump(merged_contents, f, indent=4)

    with open(file_path, 'r') as f:
        data = json.load(f)
        
    print(len(data))

    average_tokens = calculate_average_tokens(data)
    
    # Display the average token counts per chapter
    print(f"Average # tokens: {average_tokens}")
    
if __name__ == "__main__":
    main()

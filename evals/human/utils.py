from collections import defaultdict
import json
import random
import os

def read_json(file_path):
    """Utility to read a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def write_json(data, file_path):
    """Utility to write data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def find_responses(first_file, second_file, output_file):
    # Load data from both JSON files
    exercises = read_json(first_file)
    detailed_responses = read_json(second_file)
    
    # Flatten the exercise queries into a list with added type info
    flattened_queries = []
    for exercise_type, queries in exercises.items():
        for query in queries:
            query['exercise_type'] = exercise_type  # Add type to ease matching
            flattened_queries.append(query)
    
    # Mapping from query string to desired data structure
    query_to_response = {item['query']: item for item in flattened_queries}

    print(query_to_response)

    # Prepare the results
    results = []
    for entry in detailed_responses:
        query_text = entry['query']['query']
        if query_text in query_to_response:
            results.append({
                'query': query_text,
                'human_response': query_to_response[query_text]['human_response'],
                'response': entry['response']['text'],
                'exercise_format': entry['query']['requested_exercise_format'],
                'topic': entry['query']['topic']
            })
    
    # Save results to a new JSON file
    write_json(results, output_file)
    print(f"Results written to {output_file}")


def merge_and_shuffle_json_files(file_paths, output_file_path):
    combined_data = []
    
    # Read each file and extend the combined_data list
    for file_path in file_paths:
        data = read_json(file_path)
        for item in data:
            item['source_file'] = os.path.basename(file_path)  # Add source file name
        combined_data.extend(data)
    
    # Shuffle the combined data
    random.shuffle(combined_data)
    
    # Write the shuffled data to a new JSON file
    write_json(combined_data, output_file_path)
    print(f"Merged and shuffled data written to {output_file_path}")

# Function to filter out excluded queries
def filter_excluded(data, excluded):
    excluded_queries = set(
        item['query'] for key in excluded for item in excluded[key]
    )
    filtered_data = {
        'short-answer': [item for item in data if item['requested_exercise_format'] == 'short-answer' and item['query'] not in excluded_queries],
        'true-false': [item for item in data if item['requested_exercise_format'] == 'true-false' and item['query'] not in excluded_queries],
        'long-answer': [item for item in data if item['requested_exercise_format'] == 'long-answer' and item['query'] not in excluded_queries]
    }
    return filtered_data

def distribute_unique_queries(data, num_files):
    """Distributes unique queries into specified number of files."""
    grouped_queries = defaultdict(list)

    # Group queries by their 'query' text
    for item in data:
        grouped_queries[item['query']].append(item)

    # Prepare the output lists
    output_lists = [[] for _ in range(num_files)]

    # Distribute the groups across the files
    for queries in grouped_queries.values():
        random.shuffle(queries)  # Shuffle to ensure random distribution
        for i, query in enumerate(queries):
            output_lists[i % num_files].append(query)

    return output_lists


def split_and_save_json(data, base_output_path, num_splits=3):
    # Distribute unique queries across the splits
    distributed_data = distribute_unique_queries(data, num_splits)
    
    # Write each split to a separate file
    for i, split_data in enumerate(distributed_data):
        output_file_path = f"{base_output_path}-part-{i + 1}.json"
        write_json(split_data, output_file_path)
        print(f"Data written to {output_file_path}")
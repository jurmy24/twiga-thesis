import json
import os
from typing import List
from dotenv import load_dotenv
import csv

from src.models import PipelineData
from src.utils import load_json_to_pipelinedata

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR_PATH")

def extract_eval_data(file_path: str) -> List[PipelineData]:

    with open(file_path, 'r') as file:
        data = json.load(file)

    return load_json_to_pipelinedata(data)

# Function to append a string to a text file
def append_to_file(file_path: str, text: str):
    with open(file_path, 'a') as file:  # Open file in append mode
        file.write(text + '\n')  # Append text followed by a newline

def save_tuples_to_csv(file_path: str, data: list):
    """
    Save a list of tuples to a CSV file.

    :param file_path: Path of the CSV file where data will be saved
    :param data: List of tuples containing the data
    """
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
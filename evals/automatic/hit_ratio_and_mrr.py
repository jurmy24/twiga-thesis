from typing import List, Dict, Tuple
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm

from src.llms.openai_requests import instructor_openai_request
from src.models import PipelineData
from evals.automatic.models import HitResponse
from evals.automatic.test_utils import append_to_file, save_tuples_to_csv

load_dotenv
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR_PATH")

HR_SYSTEM_PROMPT = """
    You are an expert Tanzanian Geography education system tool that strictly responds with either a 1 or 0.
    A 1 means that the provided retrieved document provides highly relevant information for the topic requested in the user query. 
    It does not need to be in the same format. I.e. if the user requests a question or exercise to be generated about a certain topic classify it as 1 as long as the retrieved document is about the topic they mentioned.
    A 0 means that the provided retrieved document is not related to the topic of the query.
    Respond immediately with either a 1 or a 0. Don't explain why.
"""

HR_USER_PROMPT = ("User Query: {query}\nRetrieved Document: {doc}")

# Function to calculate hit ratio for a list of PipelineData objects
def compute_hit_ratio_and_mrr(pipeline_data_list: List[PipelineData]) -> Tuple[List[Tuple], float, float]:
    # TO REMOVE:
    # pipeline_data_list = pipeline_data_list[0:5]
    first_results = [("Query", "First Retrieved Document", "GPT-4-1106-preview Score on Relevancy")]

    hits = 0
    reciprocal_rank_sum = 0
    for data in tqdm(pipeline_data_list, desc="Going through pipeline data"):
        query = data.query.query
        retrieved_content_docs = [doc.source.chunk for doc in data.retrieved_docs if doc.source.metadata.doc_type == "Content"]

        hit = False
        reciprocal_rank = 0
        for i, doc in enumerate(retrieved_content_docs):
            i += 1 # so that i starts at 1
            prompt = HR_USER_PROMPT.format(query=query, doc=doc)

            score_data = instructor_openai_request(
                model="gpt-4-1106-preview",
                response_model=HitResponse,
                messages=[
                    {"role": "system", "content": HR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ], 
                max_tokens=10,
                temperature=0
            )

            if i == 1:
                first_results.append((query, doc, score_data.hit))
            
            if score_data.hit != 0:
                hit = True
                reciprocal_rank = 1/i
                break # break out of for-loop in order not to do unnecessary OpenAI calls
        
        reciprocal_rank_sum += reciprocal_rank
        if hit:
            hits += 1

    len_data = len(pipeline_data_list)
    hit_rate = hits / len_data
    mean_reciprocal_rank = reciprocal_rank_sum / len_data

    return first_results, hit_rate, mean_reciprocal_rank

# Example usage:
if __name__ == "__main__":

    from evals.automatic.test_utils import extract_eval_data

    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR_PATH")

    data_file = os.path.join(DATA_DIR, "results", "5-pipeline-gpt-3-5.json")
    results_file = os.path.join(DATA_DIR, "results", "results.txt")
    csv_file = os.path.join(DATA_DIR, "results", "pipeline-5-7-hit-analysis2.csv")

    pipeline_data = extract_eval_data(data_file)


    # Compute hit ratio with a threshold of 0.5
    first_results, hit_rate, mrr = compute_hit_ratio_and_mrr(pipeline_data)

    save_tuples_to_csv(csv_file, first_results)
    append_to_file(results_file, f"Pipeline (5-7) Hit Rate: {hit_rate}")
    append_to_file(results_file, f"Pipeline (5-7) MRR: {mrr}")
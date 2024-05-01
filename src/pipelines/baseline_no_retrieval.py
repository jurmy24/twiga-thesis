import json
from dotenv import load_dotenv
import os
from typing import List, Literal, Tuple
from tqdm import tqdm
import logging

from openai.types.chat import ChatCompletion
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer

from src.llms.groq_requests import groq_request
from src.llms.openai_requests import openai_request
from src.models import EvalQuery, PipelineData, ResponseSchema
from src.prompt_templates import BASELINE_GENERATOR_PROMPT
from src.utils import get_embedding, load_json_to_evalquery, save_objects_as_json

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR_PATH")

"""
This is the most basic implementation of the tool
"""
def baseline_generator(query: EvalQuery, model: Literal["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09","llama3-70b-8192"]) -> Tuple[EvalQuery, str]:
    try:        
        messages = [
            {"role": BASELINE_GENERATOR_PROMPT.role, "content": BASELINE_GENERATOR_PROMPT.content},
            {"role": "user", "content": query.query}
        ]

        if model == "llama3-70b-8192":
            res = groq_request(
                llm=model,
                verbose=False,
                messages=messages, 
                max_tokens=100,
            )
        else:
            res: ChatCompletion = openai_request(
                model=model,
                verbose=False,
                messages=messages,
                max_tokens=100,  # Adjust based on the expected length of the enhanced query
            )

        # Extract the enhanced query text from the response
        gen_query = res.choices[0].message.content

        return query, gen_query
    
    except Exception as e:
        logger.error(f"An error occurred when generating a baseline response query: {e}")
        return None
    
def run_baseline_fast(queries: List[EvalQuery], model: Literal["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09","llama3-70b-8192"]) -> List[PipelineData]:
    embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    pipe_data: List[PipelineData] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(baseline_generator, query, model) for query in queries]
        for future in tqdm(as_completed(futures), total=len(queries), desc="Generating queries"):
            q, res = future.result()
            if q and res:
                # Compute the embeddings
                q.embedding = get_embedding(q.query, embedding_model)
                res_embedding = get_embedding(res, embedding_model)

                res_data: ResponseSchema = ResponseSchema(text=res, embedding=res_embedding)
                pipe_result: PipelineData = PipelineData(query=q, response=res_data)
                pipe_data.append(pipe_result)

    return pipe_data

def run_baseline_slow(queries: List[EvalQuery], model: Literal["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09","llama3-70b-8192"]) -> List[PipelineData]:
    embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    pipe_data: List[PipelineData] = []
    for q in tqdm(queries, desc="Generating queries"):
        query, res = baseline_generator(q, model)

        if query and res:
            # Compute the embeddings
            query.embedding = get_embedding(query.query, embedding_model)
            res_embedding = get_embedding(res, embedding_model)

            res_data: ResponseSchema = ResponseSchema(text=res, embedding=res_embedding)
            pipe_result: PipelineData = PipelineData(query=query, response=res_data)
            pipe_data.append(pipe_result)

    return pipe_data

if __name__ == "__main__":
    test_path = os.path.join(DATA_DIR, "datasets", "test-prompts.json")
    test_path_control = os.path.join(DATA_DIR, "datasets", "control-test-prompts.json")
    save_path = os.path.join(DATA_DIR, "results", "3-baseline-llama3.json")
    save_path_control = os.path.join(DATA_DIR, "results", "3-baseline-llama3-control.json")

    with open(test_path, 'r') as file:
        data = json.load(file)

    eval_queries = load_json_to_evalquery(data)

    generated_queries = run_baseline_slow(eval_queries, model="llama3-70b-8192")

    # Save to JSON
    save_objects_as_json(generated_queries, save_path, rewrite=True)

import json
import os
from typing import List, Tuple
from dotenv import load_dotenv
import logging
import openai

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llms.openai_requests import openai_assistant_request
from src.models import EvalQuery, PipelineData, ResponseSchema
from src.prompt_templates import ASSISTANT_OPENAI_PROMPT
from sentence_transformers import SentenceTransformer
from src.utils import get_embedding, load_json_to_evalquery, save_objects_as_json

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING) # to get rid of the httpx logs

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG = os.getenv("OPENAI_ORG")
TWIGA_ASSISTANT_ID = os.getenv("TWIGA_OPENAI_ASSISTANT_ID")

"""
This is the openai assistants implementation of the tool
"""

def create_assistant_and_knowledge_base(client: openai.OpenAI, ass_name: str, openai_model: str, system_prompt: str, data_path_list: List[str]):
    # Create the assistant
    client.beta.assistants.create(
        name=ass_name,
        instructions=system_prompt,
        model=openai_model,
        tools=[{"type": "file_search"}],
    )

    # Create a vector store
    vector_store = client.beta.vector_stores.create(name="Twiga Documents")

    # Ready the files for upload to OpenAI 
    file_paths = data_path_list
    file_streams = [open(path, "rb") for path in file_paths]
    
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,and poll the status of the file batch for completion.
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )
    
    # You can print the status and the file counts of the batch to see the result of this operation. 
    print(file_batch.status)
    print(file_batch.file_counts)

    assistant = client.beta.assistants.update(
      assistant_id=assistant.id,
      tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

def assistant_generator(query: EvalQuery, client, assistant) -> Tuple[EvalQuery, str, bool]:
    try: 
        params = {
            "messages": [
                {
                "role": "user",
                "content": query.query,
                }
            ]
        }
        
        message_content = openai_assistant_request(client, assistant, verbose=False, **params)

        # The below stuff is solely for the use of citations
        annotations = message_content.annotations
        invoked_file_search = False
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
            if getattr(annotation, "file_citation", None):
                invoked_file_search = True
        
        return (query, message_content.value, invoked_file_search)
    except Exception as e:
        logger.error(f"An error occurred when generating an assistant response query: {e}")
        return None, None, None

def run_assistant_fast(queries: List[EvalQuery], client, assistant) -> List[PipelineData]:
    embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    pipe_data: List[PipelineData] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(assistant_generator, query, client, assistant) for query in queries]
        for future in tqdm(as_completed(futures), total=len(queries), desc="Generating queries"):
            q, res, invoked_file_search = future.result()
            if q and res:
                # Compute the embeddings
                q.embedding = get_embedding(q.query, embedding_model)
                res_embedding = get_embedding(res, embedding_model)

                res_data: ResponseSchema = ResponseSchema(text=res, embedding=res_embedding, invoked_file_search=invoked_file_search)
                pipe_result: PipelineData = PipelineData(query=q, response=res_data)
                pipe_data.append(pipe_result)

    return pipe_data 
    
def run_assistant_slow(queries: List[EvalQuery], client, assistant) -> List[PipelineData]:
    embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    pipe_data: List[PipelineData] = []
    for q in tqdm(queries, desc="Generating queries"):
        query, res, invoked_file_search = assistant_generator(q, client, assistant)
        if query and res:
            # Compute the embeddings
            query.embedding = get_embedding(query.query, embedding_model)
            res_embedding = get_embedding(res, embedding_model)

            res_data: ResponseSchema = ResponseSchema(text=res, embedding=res_embedding, invoked_file_search=invoked_file_search)
            pipe_result: PipelineData = PipelineData(query=query, response=res_data)
            pipe_data.append(pipe_result)

    return pipe_data

if __name__ == "__main__":
    # Create assistant and knowledge base for geography teacher (note: already have one)
    client = openai.OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG)
    """
    ass_name = "Tanzania Secondary School Geography Exercise/Question Generator"
    openai_model = "gpt-3.5-turbo-0125"
    system_prompt = ASSISTANT_OPENAI_PROMPT
    file_path = os.path.join(DATA_DIR, "documents", "pdf", "GEOGRAPHY-F2-TIE.pdf")
    create_assistant_and_knowledge_base(client, ass_name, openai_model, system_prompt, [file_path])
    """
    # Retrieve the Assistant
    assistant = client.beta.assistants.retrieve(TWIGA_ASSISTANT_ID)

    test_path = os.path.join(DATA_DIR, "datasets", "test-prompts.json")
    test_path_control = os.path.join(DATA_DIR, "datasets", "control-test-prompts.json")
    save_path = os.path.join(DATA_DIR, "results", "4-assistant-gpt-3-5.json")
    save_path_control = os.path.join(DATA_DIR, "results", "4-assistant-gpt-3-5-control.json")

    test_path = os.path.join(DATA_DIR, "datasets", "crap.json")

    with open(test_path, 'r') as file:
        data = json.load(file)

    eval_queries = load_json_to_evalquery(data)

    generated_queries = run_assistant_slow(eval_queries, client, assistant)

    # Save to JSON
    save_objects_as_json(generated_queries, save_path, rewrite=False)
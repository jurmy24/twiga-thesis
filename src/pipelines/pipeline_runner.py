import json
from typing import List, Literal
from src.DataSearch import DataSearch
from src.llms.groq_requests import groq_request
from src.llms.openai_requests import openai_request
from src.models import EvalQuery, ResponseSchema, RetrievedDocSchema, RewrittenQuery, PipelineData
from src.prompt_templates import PIPELINE_QUESTION_GENERATOR_PROMPT, PIPELINE_QUESTION_GENERATOR_USER_PROMPT
from src.utils import load_json_to_evalquery, get_embedding, load_json_to_pipelinedata, save_objects_as_json
from src.pipelines.modules import elasticsearch_retriever, query_rewriter, rerank
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR_PATH")

def process_queries(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
    queries: List[EvalQuery] = load_json_to_evalquery(data)

    results: List[EvalQuery] = []
    for eval_query in tqdm(queries, desc="Rewriting queries..."):

        # Put the query into the query rewriter
        original_query = eval_query.query
        rewritten_query = query_rewriter(original_query, llm="llama3-8b-8192") # would be better if this could be done as a batch but I guess not...

        # Get the embeddings of both the original query and the rewritten one
        original_embedding = get_embedding(original_query, model)
        rewritten_embedding = get_embedding(rewritten_query, model)

        rwq: RewrittenQuery = RewrittenQuery(rewritten_query_str=rewritten_query, embedding=rewritten_embedding)

        evq: EvalQuery = EvalQuery(
            query=original_query,
            requested_exercise_format=eval_query.requested_exercise_format,
            topic=eval_query.topic,
            embedding=original_embedding,
            rewritten_query=rwq
            )
        
        pip_data: PipelineData = PipelineData(query=evq)
        
        results.append(pip_data)

    return results

def process_rewritten_queries(file_path: str) -> List[PipelineData]:
    with open(file_path, 'r') as file:
        data = json.load(file)

    pipe_data: List[PipelineData] = load_json_to_pipelinedata(data)

    results: List[PipelineData] = []
    model_class = DataSearch()
    for item in tqdm(pipe_data, desc="Retrieving documents"):

        # Fetch the rewritten query to search on
        # original_query = item.query.query
        rewritten_query = item.query.rewritten_query.rewritten_query_str
        # item.query.rewritten_query = None

        retrieved_content: List[RetrievedDocSchema] = elasticsearch_retriever(model_class=model_class, retrieval_msg=rewritten_query, size=5, doc_type="Content", retrieve_dense=True, retrieve_sparse=True)
        retrieved_exercises: List[RetrievedDocSchema] = elasticsearch_retriever(model_class=model_class, retrieval_msg=rewritten_query, size=2, doc_type="Exercise", retrieve_dense=True, retrieve_sparse=True)

        # retrieved_content = rerank(item.query, retrieved_content, num_results=5)
        # retrieved_exercises = rerank(item.query, retrieved_exercises, num_results=2)

        retrieved_docs = retrieved_content + retrieved_exercises

        pip_data: PipelineData = PipelineData(query=item.query, retrieved_docs=retrieved_docs)
        
        results.append(pip_data)

    return results

def pipeline_generator(prompt: str, query: str, model: Literal["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09","llama3-70b-8192"], verbose:bool = False) -> str:
    try:    
        # TODO: Check if this is the right format for the prompting!    
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]

        if verbose:
            logger.info(f"--------------------------")
            logger.info(f"System prompt: \n{prompt}")
            logger.info(f"--------------------------")
            logger.info(f"User prompt: \n{query}")


        if model == "llama3-70b-8192":
            res = groq_request(
                llm=model,
                verbose=False,
                messages=messages, 
                max_tokens=100,
            )
        else:
            res = openai_request(
                model=model,
                verbose=False,
                messages=messages,
                max_tokens=100,  # Adjust based on the expected length of the enhanced query
            )

        # Extract the enhanced query text from the response
        gen_query = res.choices[0].message.content

        return gen_query
    
    except Exception as e:
        logger.error(f"An error occurred when generating a response query: {e}")
        return None

def run_data_through_generator(pipe_data: List[PipelineData], model: Literal["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09","llama3-70b-8192"], verbose: bool=False) -> List[PipelineData]:
    results: List[PipelineData] = []
    embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    for item in tqdm(pipe_data, desc="Retrieving documents"):

        retrieved_docs = item.retrieved_docs

        retrieved_content = [doc for doc in retrieved_docs if doc.source.metadata.doc_type == "Content"]
        retrieved_exercise = [doc for doc in retrieved_docs if doc.source.metadata.doc_type == "Exercise"]
        
        # TODO: Put in a check to make sure these exist

        # Format the context
        context_parts = []

        context_parts.append(f"### Context from the textbook ({retrieved_content[0].source.metadata.title})\n")
        for index, doc in enumerate(retrieved_content):

            metadata = doc.source.metadata
            if metadata.chapter and metadata.subsection and metadata.subsubsection:
                heading = f"-Chunk number {index} from chapter {metadata.chapter}, subsection {metadata.subsection}, subsubsection {metadata.subsubsection}"
            elif metadata.chapter and metadata.subsection:
                heading = f"-Chunk number {index} from chapter {metadata.chapter}, subsection {metadata.subsection}"
            elif metadata.chapter:
                heading = f"-Chunk number {index} from chapter {metadata.chapter}"
            else:
                heading = f"-Chunk number {index}"
            
            context_parts.append(heading)
            context_parts.append(f"{doc.source.chunk}")

        context_parts.append(f"\n### Sample exercises from the textbook ({retrieved_content[0].source.metadata.title})\n")
        for doc in retrieved_exercise:
            metadata = doc.source.metadata
            if metadata.chapter and metadata.subsection and metadata.subsubsection:
                heading = f"-Exercise of type {metadata.exercise_format} from chapter {metadata.chapter}, subsection {metadata.subsection}, subsubsection {metadata.subsubsection}"
            elif metadata.chapter and metadata.subsection:
                heading = f"-Exercise of type {metadata.exercise_format} from chapter {metadata.chapter}, subsection {metadata.subsection}"
            elif metadata.chapter:
                heading = f"-Exercise of type {metadata.exercise_format} from chapter {metadata.chapter}"
            else:
                heading = f"-Exercise of type {metadata.exercise_format}"
    
            context_parts.append(heading)
            context_parts.append(f"{doc.source.chunk}")

        context = "\n".join(context_parts)
    
        system_prompt = PIPELINE_QUESTION_GENERATOR_PROMPT.format()
        user_prompt = PIPELINE_QUESTION_GENERATOR_USER_PROMPT.format(query=item.query.query, context_str=context)

        res = pipeline_generator(system_prompt, user_prompt, model, verbose=verbose)

        res_proper: ResponseSchema = ResponseSchema(text=res, embedding=get_embedding(res, embedding_model))

        pip_data: PipelineData = PipelineData(query=item.query, retrieved_docs=retrieved_docs, response=res_proper)
        
        results.append(pip_data)

    return results

if __name__ == "__main__":
    
    input_file = os.path.join(DATA_DIR, "datasets", "test-prompts-rewritten-retrieved.json")
    # output_file = os.path.join(DATA_DIR, "results", "6-pipeline-gpt-4.json")
    output_file = os.path.join(DATA_DIR, "results", "7-pipeline-llama3.json")

    with open(input_file, 'r') as file:
        data = json.load(file)

    incomplete_pipeline_data = load_json_to_pipelinedata(data)

    res = run_data_through_generator(incomplete_pipeline_data, "llama3-70b-8192", verbose=False)

    save_objects_as_json(res, output_file, rewrite=True)

    # res = process_rewritten_queries(input_file)

    # save_objects_as_json(res, output_file, rewrite=True)






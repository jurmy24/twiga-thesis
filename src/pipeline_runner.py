import json
from typing import List, Literal
from src.DataSearch import DataSearch
from src.models import EvalQuery, RetrievedDocSchema, RewrittenQuery, PipelineData
from src.utils import load_json_to_evalquery, get_embedding, load_json_to_pipelinedata, save_objects_as_json
from src.pipelines.modules import elasticsearch_retriever, query_rewriter, rerank
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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

        
        retrieved_content: List[RetrievedDocSchema] = elasticsearch_retriever(model_class=model_class, retrieval_msg=rewritten_query, size=10, doc_type="Content", retrieve_dense=True, retrieve_sparse=True)
        retrieved_exercises: List[RetrievedDocSchema] = elasticsearch_retriever(model_class=model_class, retrieval_msg=rewritten_query, size=5, doc_type="Exercise", retrieve_dense=True, retrieve_sparse=True)

        retrieved_content = rerank(item.query, retrieved_content, num_results=5)
        retrieved_exercises = rerank(item.query, retrieved_exercises, num_results=2)

        retrieved_docs = retrieved_content + retrieved_exercises

        pip_data: PipelineData = PipelineData(query=item.query, retrieved_docs=retrieved_docs)
        
        results.append(pip_data)

    return results

if __name__ == "__main__":
    
    # test_prompts_file = os.path.join(DATA_DIR, "datasets", "test-prompts.json")
    # test_prompts_rewritten_file = os.path.join(DATA_DIR, "datasets", "test-prompts-rewritten.json")
    # control_test_prompts_file = os.path.join(DATA_DIR, "datasets", "control-test-prompts.json")
    # control_test_prompts_rewritten_file = os.path.join(DATA_DIR, "datasets", "control-test-prompts-rewritten.json")
    # results = process_queries(control_test_prompts_file)
    # save_objects_as_json(results, control_test_prompts_rewritten_file, rewrite=True)

    test_prompts_rewritten_file = os.path.join(DATA_DIR, "datasets", "test-prompts-rewritten.json")
    test_prompts_retrieved_file = os.path.join(DATA_DIR, "datasets", "test-prompts-rewritten-retrieved.json")

    res = process_rewritten_queries(test_prompts_rewritten_file)

    save_objects_as_json(res, test_prompts_retrieved_file, rewrite=True)






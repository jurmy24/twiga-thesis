import json
from typing import List, Literal
from src.models import EvalQuery, RewrittenQuery
from src.utils import load_json_to_evalquery, get_embedding, save_objects_as_json
from src.pipelines.modules import query_rewriter
import os
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR_PATH")

def process_queries(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    queries: List[EvalQuery] = load_json_to_evalquery(data)

    results: List[EvalQuery] = []
    for eval_query in queries:

        # Put the query into the query rewriter
        original_query = eval_query.query
        rewritten_query = query_rewriter(original_query, llm="llama3-8b-8192") # would be better if this could be done as a batch but I guess not...

        # Get the embeddings of both the original query and the rewritten one
        original_embedding = get_embedding(original_query)
        rewritten_embedding = get_embedding(rewritten_query)

        rwq: RewrittenQuery = RewrittenQuery(rewritten_query_str=rewritten_query, embedding=rewritten_embedding)

        evq: EvalQuery = EvalQuery(
            query=original_query,
            requested_exercise_format=eval_query.requested_exercise_format,
            topic=eval_query.topic,
            embedding=original_embedding,
            rewritten_query=rwq
            )
        
        results.append(evq)

    return results

if __name__ == "__main__":
    
    test_prompts_file = os.path.join(DATA_DIR, "datasets", "test-prompts.json")
    test_prompts_rewritten_file = os.path.join(DATA_DIR, "datasets", "test-prompts-rewritten.json")
    results = process_queries(test_prompts_file)
    save_objects_as_json(results, test_prompts_rewritten_file)





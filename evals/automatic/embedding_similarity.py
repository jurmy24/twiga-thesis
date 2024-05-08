from tqdm import tqdm
from src.models import PipelineData
from typing import List, Tuple
import numpy as np

def cosine_similarity_pipeline_from_stored_embeddings(pipeline_data_list: List[PipelineData]):
    """
    Compute cosine similarity for each response in the PipelineData list using stored embeddings and return CSV data.
    """
    csv_data = [("Generated Query", "First Content Document", "Content Cosine Similarity", "First Exercise Document", "Exercise Cosine Similarity")]

    avg_content_similarity = 0
    avg_exercise_similarity = 0
    for data in tqdm(pipeline_data_list, desc="Generating cosine similarities..."):
        generated_question_embedding = np.array(data.response.embedding)

        retrieved_content_embeddings = [(doc.source.chunk, np.array(doc.source.embedding)) for doc in data.retrieved_docs if doc.source.metadata.doc_type == "Content"]
        retrieved_exercise_embeddings = [(doc.source.chunk, np.array(doc.source.embedding)) for doc in data.retrieved_docs if doc.source.metadata.doc_type == "Exercise"]

        # Compute average similarity for content documents
        if retrieved_content_embeddings:
            content_similarities = [
                np.dot(generated_question_embedding, embedding) /
                (np.linalg.norm(generated_question_embedding) * np.linalg.norm(embedding))
                for _, embedding in retrieved_content_embeddings
            ]
            mini_avg_content_similarity = sum(content_similarities) / len(content_similarities)

        # Compute average similarity for exercise documents
        if retrieved_exercise_embeddings:
            exercise_similarities = [
                np.dot(generated_question_embedding, embedding) /
                (np.linalg.norm(generated_question_embedding) * np.linalg.norm(embedding))
                for _, embedding in retrieved_exercise_embeddings
            ]
            mini_avg_exercise_similarity = sum(exercise_similarities) / len(exercise_similarities)

        avg_content_similarity += mini_avg_content_similarity
        avg_exercise_similarity += mini_avg_exercise_similarity

        csv_data.append((
            data.response.text,
            retrieved_content_embeddings[0],
            mini_avg_content_similarity,
            retrieved_exercise_embeddings[0],
            mini_avg_exercise_similarity
        ))

    avg_content_similarity = avg_content_similarity / len(pipeline_data_list)
    avg_exercise_similarity = avg_exercise_similarity / len(pipeline_data_list)

    return csv_data, avg_content_similarity, avg_exercise_similarity

if __name__ == "__main__":
    from evals.automatic.test_utils import extract_eval_data, append_to_file, save_tuples_to_csv
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR_PATH")

    data_file = os.path.join(DATA_DIR, "results", "7-pipeline-llama3.json")
    results_file = os.path.join(DATA_DIR, "results", "results.txt")
    csv_file_cosine = os.path.join(DATA_DIR, "results", "pipeline-7-all-MiniLM-l6-v2-cosine-analysis.csv")

    pipeline_data = extract_eval_data(data_file)

    csv_cosine_data, similarity_content, similarity_exercise = cosine_similarity_pipeline_from_stored_embeddings(pipeline_data)
    save_tuples_to_csv(csv_file_cosine, csv_cosine_data)
    append_to_file(results_file, f"Pipeline (7) all-MiniLM-l6-v2 Cosine Content Similarity: {similarity_content}")
    append_to_file(results_file, f"Pipeline (7) all-MiniLM-l6-v2 BERT Cosine Exercise Similarity: {similarity_exercise}")
from typing import Tuple, List
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer
import numpy as np

from src.models import PipelineData

def compute_bertscore(scorer: BERTScorer, reference, candidate) -> Tuple[float, float, float]:    
    P, R, F1 = scorer.score([candidate], [reference])
    return (P.mean(), R.mean(), F1.mean())

def bertscore_computation_pipeline(pipeline_data_list: List[PipelineData], scorer: BERTScorer):
    """
    Compute BERTScore for each response in the PipelineData list and save results to a CSV file.
    """
    csv_data = [("Generated Query", "First Document", "Mean Content Precision", "Mean Content Recall", "Mean Content F1", "First Exercise", "Mean Exercise Precision", "Mean Exercise Recall", "Mean Exercise F1")]

    average_P_content, average_R_content, average_F1_content = 0.0, 0.0, 0.0
    average_P_exercise, average_R_exercise, average_F1_exercise = 0.0, 0.0, 0.0
    for data in tqdm(pipeline_data_list, desc="Processing data from pipeline..."):

        generated_question = data.response.text

        retrieved_content_docs = [doc.source.chunk for doc in data.retrieved_docs if doc.source.metadata.doc_type == "Content"]
        retrieved_exercise_docs = [doc.source.chunk for doc in data.retrieved_docs if doc.source.metadata.doc_type == "Exercise"]

        mini_avg_P_content, mini_avg_R_content, mini_avg_F1_content = 0.0, 0.0, 0.0
        for doc in retrieved_content_docs:
            P, R, F1 = compute_bertscore(scorer, generated_question, doc)
            mini_avg_P_content += P
            mini_avg_R_content += R
            mini_avg_F1_content += F1
        
        mini_avg_P_content = mini_avg_P_content / len(retrieved_content_docs)
        mini_avg_R_content = mini_avg_R_content / len(retrieved_content_docs)
        mini_avg_F1_content = mini_avg_F1_content / len(retrieved_content_docs)

        mini_avg_P_exercise, mini_avg_R_exercise, mini_avg_F1_exercise = 0.0, 0.0, 0.0
        for doc in retrieved_exercise_docs:
            P, R, F1 = compute_bertscore(scorer, generated_question, doc)
            mini_avg_P_exercise += P
            mini_avg_R_exercise += R
            mini_avg_F1_exercise += F1
        
        mini_avg_P_exercise = mini_avg_P_exercise / len(retrieved_exercise_docs)
        mini_avg_R_exercise = mini_avg_R_exercise / len(retrieved_exercise_docs)
        mini_avg_F1_exercise = mini_avg_F1_exercise / len(retrieved_exercise_docs)
        
        csv_data.append((generated_question, retrieved_content_docs[0], mini_avg_P_content, mini_avg_R_content, mini_avg_F1_content, retrieved_exercise_docs[0], mini_avg_P_exercise, mini_avg_R_exercise, mini_avg_F1_exercise))

        average_P_content += mini_avg_P_content
        average_R_content += mini_avg_R_content
        average_F1_content += mini_avg_F1_content
        average_P_exercise += mini_avg_P_exercise
        average_R_exercise += mini_avg_R_exercise
        average_F1_exercise += mini_avg_F1_exercise
    
    len_data = len(pipeline_data_list)
    average_P_content = average_P_content / len_data
    average_R_content = average_R_content / len_data
    average_F1_content = average_F1_content / len_data
    average_P_exercise = average_P_exercise / len_data
    average_R_exercise = average_R_exercise / len_data
    average_F1_exercise = average_F1_exercise / len_data

    return csv_data, average_P_content, average_R_content, average_F1_content, average_P_exercise, average_R_exercise, average_F1_exercise

if __name__ == "__main__":
    from evals.automatic.test_utils import extract_eval_data, append_to_file, save_tuples_to_csv
    import os
    from dotenv import load_dotenv

    # Bert tokenizer and model for cosine similarity computation
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # BERTScore scorer
    scorer = BERTScorer(model_type='bert-base-uncased')

    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR_PATH")

    data_file = os.path.join(DATA_DIR, "complete_runs", "5-pipeline-gpt.json")
    results_file = os.path.join(DATA_DIR, "complete_runs", "results.txt")
    csv_file = os.path.join(DATA_DIR, "complete_runs", "pipeline-7-BERT-analysis.csv")
    csv_file_cosine = os.path.join(DATA_DIR, "complete_runs", "pipeline-7-BERT-cosine-analysis.csv")

    pipeline_data = extract_eval_data(data_file)

    csv_data, P_content, R_content, F1_content, P_exercise, R_exercise, F1_exercise = bertscore_computation_pipeline(pipeline_data, scorer)
    
    save_tuples_to_csv(csv_file, csv_data)
    append_to_file(results_file, f"Pipeline (7) BERTScore Content Precision: {P_content}, Recall: {R_content}, F1: {F1_content}")
    append_to_file(results_file, f"Pipeline (7) BERTScore Exercise Precision: {P_exercise}, Recall: {R_exercise}, F1: {F1_exercise}")

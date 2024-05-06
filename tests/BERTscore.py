from typing import Tuple, List
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
import numpy as np
import csv

from src.models import PipelineData

def compute_bertscore(scorer: BERTScorer, reference, candidate) -> Tuple[float, float, float]:    
    P, R, F1 = scorer.score([candidate], [reference])
    return (P.mean(), R.mean(), F1.mean())

def compute_bert_cosine(tokenizer, model, text1: str, text2: str) -> float:
    # Step 4: Prepare the texts for BERT
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    # Step 5: Feed the texts to the BERT model
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

    # Step 6: Obtain the representation vectors
    embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()
    embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()

    # Step 7: Calculate cosine similarity
    similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))

    # Step 8: Print the result
    return similarity[0][0]

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


def bert_cosine_similarity_pipeline(pipeline_data_list: List[PipelineData], tokenizer: BertTokenizer, model: BertModel):
    """
    Compute cosine similarity for each response in the PipelineData list and save results to a CSV file.
    """
    csv_data = [("Generated Query", "First Content Document", "Content Cosine Similarity", "First Exercise Document", "Exercise Cosine Similarity")]

    avg_content_similarity = 0
    avg_exercise_similarity = 0
    for data in tqdm(pipeline_data_list, desc="Generating cosine similarities..."):
        generated_question = data.response.text

        inputs1 = tokenizer(generated_question, return_tensors="pt", padding=True, truncation=True)
        outputs1 = model(**inputs1)
        embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()

        retrieved_content_docs = [doc.source.chunk for doc in data.retrieved_docs if doc.source.metadata.doc_type == "Content"]
        retrieved_exercise_docs = [doc.source.chunk for doc in data.retrieved_docs if doc.source.metadata.doc_type == "Exercise"]

        mini_avg_content_similarity = 0.0
        for doc in retrieved_content_docs:
            inputs2 = tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
            outputs2 = model(**inputs2)
            embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()

            similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
            similarity = similarity[0][0]
            mini_avg_content_similarity += similarity
        
        mini_avg_content_similarity = mini_avg_content_similarity / len(retrieved_content_docs)

        mini_avg_exercise_similarity = 0.0
        for doc in retrieved_exercise_docs:
            inputs2 = tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
            outputs2 = model(**inputs2)
            embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()

            similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
            similarity = similarity[0][0]
            mini_avg_exercise_similarity += similarity
        
        mini_avg_exercise_similarity = mini_avg_exercise_similarity / len(retrieved_content_docs)

        avg_content_similarity += mini_avg_content_similarity
        avg_exercise_similarity += mini_avg_exercise_similarity

        csv_data.append((generated_question, retrieved_content_docs[0], mini_avg_content_similarity, retrieved_exercise_docs[0], mini_avg_exercise_similarity))
    
    avg_content_similarity = avg_content_similarity / len(pipeline_data_list)
    avg_exercise_similarity = avg_exercise_similarity / len(pipeline_data_list)
    return csv_data, avg_content_similarity, avg_exercise_similarity

if __name__ == "__main__":
    from tests.test_utils import extract_eval_data, append_to_file, save_tuples_to_csv
    import os
    from dotenv import load_dotenv

    # Bert tokenizer and model for cosine similarity computation
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # BERTScore scorer
    scorer = BERTScorer(model_type='bert-base-uncased')

    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR_PATH")

    data_file = os.path.join(DATA_DIR, "results", "7-pipeline-llama3.json")
    results_file = os.path.join(DATA_DIR, "results", "results.txt")
    csv_file = os.path.join(DATA_DIR, "results", "pipeline-7-BERT-analysis.csv")
    csv_file_cosine = os.path.join(DATA_DIR, "results", "pipeline-7-BERT-cosine-analysis.csv")

    pipeline_data = extract_eval_data(data_file)

    csv_data, P_content, R_content, F1_content, P_exercise, R_exercise, F1_exercise = bertscore_computation_pipeline(pipeline_data, scorer)
    
    save_tuples_to_csv(csv_file, csv_data)
    append_to_file(results_file, f"Pipeline (7) BERTScore Content Precision: {P_content}, Recall: {R_content}, F1: {F1_content}")
    append_to_file(results_file, f"Pipeline (7) BERTScore Exercise Precision: {P_exercise}, Recall: {R_exercise}, F1: {F1_exercise}")

    csv_cosine_data, similarity_content, similarity_exercise = bert_cosine_similarity_pipeline(pipeline_data, tokenizer, model)

    save_tuples_to_csv(csv_file_cosine, csv_cosine_data)
    append_to_file(results_file, f"Pipeline (7) BERT Cosine Content Similarity: {similarity_content}")
    append_to_file(results_file, f"Pipeline (7) BERT Cosine Exercise Similarity: {similarity_exercise}")

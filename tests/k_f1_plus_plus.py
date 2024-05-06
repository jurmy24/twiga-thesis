import functools
import re
from typing import List
from spacy.lang.en import English

from src.models import PipelineData

@functools.cache
def get_spacy_english():
    nlp = English()
    return nlp

def get_tokens(string_to_tokenize: str, lower: bool = True, remove_nonalphanumeric_tokens: bool = False) -> list[str]:
    nlp = get_spacy_english()
    doc = nlp.tokenizer(string_to_tokenize)
    if lower:
        tokens = [t.text.lower() for t in doc]
    else:
        tokens = [t.text for t in doc]
    if remove_nonalphanumeric_tokens:
        tokens = [token for token in tokens if re.match("\\w+", token)]
    return tokens

def compute_macro_f1(passages: list[str], generation: str, discount_text: str | None = None) -> float:
    """Returns the max F1 across all the passages.
    Depending on arguments, this can be Knowledge F1 or just F1.

    SQuAD paper (http://arxiv.org/abs/1606.05250):
    "This metric measures the average overlap between the prediction and ground truth answer.
    We treat the prediction and ground truth as bags of tokens, and compute their F1.
    We take the maximum F1 over all of the ground truth answers for a given question, and then average over all of the questions."

    K-F1++ (https://aclanthology.org/2023.findings-acl.60):
    "Knowledge-F1 (K-F1) ... calculates the unigram overlap between the response and a knowledge snippet K,
    providing a verbatim measure of grounding to the input source.
    We propose K-F1++, a variant of K-F1,
    that captures only the novel information in the generated response and discounts any lexical alignment to the question:
    it calculates the unigram overlap between the response and K,
    after subtracting any tokens appearing in the question from the response."
    To use K-F1++, pass in the text to ignore to discount_text.
    """
    # first create the tokenization function to use
    get_tokens = functools.partial(get_tokens, lower=True, remove_nonalphanumeric_tokens=True)
    generation_tokens = set(get_tokens(generation))
    if discount_text:
        discount_tokens = set(get_tokens(discount_text))
        generation_tokens -= discount_tokens
    n_predicted_tokens = len(generation_tokens)
    if n_predicted_tokens == 0:
        raise ValueError("Expected generation to be non-empty.")
    f1_scores = []
    for passage in passages:
        passage_tokens = set(get_tokens(passage))
        if discount_text:
            passage_tokens -= discount_tokens
        n_ground_truth_tokens = len(passage_tokens)
        if n_ground_truth_tokens == 0:
            continue
        n_correct_tokens = len(passage_tokens & generation_tokens)
        precision = n_correct_tokens / n_predicted_tokens
        recall = n_correct_tokens / n_ground_truth_tokens
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    if len(f1_scores) == 0:
        raise ValueError("No non-empty passages.")
    max_f1 = max(f1_scores)
    return max_f1

def compute_avg_kf1_score(pipeline_data_list: List[PipelineData]) -> float:
    kf1_scores_content = []
    kf1_scores_exercises = []
    for data in pipeline_data_list:
        retrieved_contents_passages = [doc.source.chunk for doc in data.retrieved_docs if doc.source.metadata.doc_type == "Content"]
        retrieved_exercises_passages = [doc.source.chunk for doc in data.retrieved_docs if doc.source.metadata.doc_type == "Exercise"]
        generation = data.response.text
        query_text = data.query.query
        try:
            kf1_score_content = compute_macro_f1(retrieved_contents_passages, generation, query_text)
            kf1_scores_content.append(kf1_score_content)
            kf1_score_exercise = compute_macro_f1(retrieved_exercises_passages, generation, query_text)
            kf1_scores_exercises.append(kf1_score_exercise)
        except ValueError:
            continue  # Skip empty or invalid data
    exercise_kf1 = sum(kf1_scores_exercises) / len(kf1_scores_exercises)
    content_kf1 = sum(kf1_scores_content) / len(kf1_scores_content)
    return content_kf1, exercise_kf1

if __name__ == "__main__":

    from tests.test_utils import extract_eval_data, append_to_file, save_tuples_to_csv
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR_PATH")

    data_file = os.path.join(DATA_DIR, "results", "5-pipeline-gpt-3-5-control.json")
    results_file = os.path.join(DATA_DIR, "results", "results.txt")

    pipeline_data_list = extract_eval_data(data_file)

    avg_kf1_score = compute_avg_kf1_score(pipeline_data_list)
    print(f"Average K-F1++ Score: {avg_kf1_score:.4f}")

    append_to_file(results_file, f"Pipeline (5)  Content Similarity: {similarity_content}")
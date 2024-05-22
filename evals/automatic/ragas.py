import logging
import os
from typing import List

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_relevancy, faithfulness

from evals.automatic.test_utils import append_to_file, extract_eval_data
from src.models import PipelineData

# Set the logging level for `httpx` to WARNING or higher
logging.getLogger("httpx").setLevel(logging.WARNING)

"""
Faithfulness - Measures the factual consistency of the answer to the context based on the question.
Context_relevancy - Measures how relevant the retrieved context is to the question, conveying the quality of the retrieval pipeline.
Answer_relevancy - Measures how relevant the output is to the query.
"""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAGAS_DO_NOT_TRACK = os.getenv("RAGAS_DO_NOT_TRACK")


def compute_ragas_results(
    dataset: Dataset,
    get_context_relevance: bool = True,
    only_answer_relevance: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    if only_answer_relevance:
        result = evaluate(
            dataset,
            metrics=[answer_relevancy],
        )
    elif get_context_relevance:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_relevancy],
        )
    else:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
        )

    df = result.to_pandas()
    return df


def compute_reverse_context_relevancies(
    dataset: Dataset, verbose: bool = False
) -> pd.DataFrame:
    """
    Note that this one takes question and context as input from the dataset, except now the question is the generated exercise instead of the user query.
    """
    result = evaluate(
        dataset,
        metrics=[context_relevancy],
    )

    df = result.to_pandas()

    new_column_names = {
        "context_relevancy": "reverse_context_relevancy",
        "question": "generated question",
    }

    # Rename the columns
    df = df.rename(columns=new_column_names)
    return df


def run_for_pipeline_5_7(
    pipeline_data: List[PipelineData], get_context_relevance: bool
) -> pd.DataFrame:
    data = {
        "question": [item.query.query for item in pipeline_data],
        "answer": [item.response.text for item in pipeline_data],
        "contexts": [
            [doc.source.chunk for doc in item.retrieved_docs] for item in pipeline_data
        ],
    }

    dataset = Dataset.from_dict(data)

    df_ragas = compute_ragas_results(dataset, get_context_relevance=False)

    data_reverse = {
        "question": [item.response.text for item in pipeline_data],
        "contexts": [
            [doc.source.chunk for doc in item.retrieved_docs] for item in pipeline_data
        ],
    }

    dataset_reverse = Dataset.from_dict(data_reverse)

    df_rcr = compute_reverse_context_relevancies(dataset_reverse)

    # Select only the desired column ('reverse_context_relevancy') from df_rcr
    df_rcr = df_rcr[["reverse_context_relevancy"]]

    """
    Here I merge the two dataframes
    """
    # Make sure that both DataFrames have the same number of rows
    assert len(df_ragas) == len(df_rcr), "DataFrames must have the same length"

    # Reset indices if needed to ensure alignment by index
    df_ragas.reset_index(drop=True, inplace=True)
    df_rcr.reset_index(drop=True, inplace=True)

    # Concatenate along columns
    merged_df = pd.concat([df_ragas, df_rcr], axis=1)

    return merged_df


def run_for_pipeline_1_4(pipeline_data: List[PipelineData]) -> pd.DataFrame:
    data = {
        "question": [item.query.query for item in pipeline_data],
        "answer": [item.response.text for item in pipeline_data],
        "contexts": [[""] for _ in pipeline_data],
    }

    dataset = Dataset.from_dict(data)

    df_ragas = compute_ragas_results(dataset, only_answer_relevance=True)

    return df_ragas


def compute_average_metrics_from_csv(df: pd.DataFrame, columns_to_average: List[str]):
    # Replace NaN with 0 and compute the mean for specified columns
    mean_values = df[columns_to_average].fillna(0).mean()

    if len(columns_to_average) == 4:
        # Accessing specific averages from the result
        average_gr = mean_values["faithfulness"]
        average_ar = mean_values["answer_relevancy"]
        average_cr = mean_values["context_relevancy"]
        average_rcr = mean_values["reverse_context_relevancy"]
        return average_gr, average_ar, average_cr, average_rcr
    elif len(columns_to_average) == 3:
        average_gr = mean_values["faithfulness"]
        average_ar = mean_values["answer_relevancy"]
        average_rcr = mean_values["reverse_context_relevancy"]
        return average_gr, average_ar, average_rcr
    elif len(columns_to_average) == 1:
        average_ar = mean_values["answer_relevancy"]
        return average_ar
    else:
        print("That doesn't make sense.")
        return None


if __name__ == "__main__":
    DATA_DIR = os.getenv("DATA_DIR_PATH")

    data_file = os.path.join(DATA_DIR, "results", "1-baseline-gpt-3-5-control.json")
    results_file = os.path.join(DATA_DIR, "results", "results.txt")
    csv_file_ragas = os.path.join(
        DATA_DIR, "results", "baseline-1-ragas-analysis-control.csv"
    )

    pipeline_data = extract_eval_data(data_file)

    # df = run_for_pipeline_5_7(extract_eval_data, get_context_relevance=False)
    df = run_for_pipeline_1_4(pipeline_data)

    # Append the DataFrame to the existing CSV file
    df.to_csv(csv_file_ragas)
    # df.to_csv(csv_file_ragas, mode='a', index=False, header=False) # set index=False

    """
    Here I compute the metric averages and store them in the results file
    """
    df = pd.read_csv(csv_file_ragas)

    # Columns to calculate the mean
    columns_to_average = ["answer_relevancy"]  # Specify the columns you want to average

    average_ar = compute_average_metrics_from_csv(df, columns_to_average)

    append_to_file(
        results_file, f"Pipeline (1) RAGAS Answer Relevancy Control: {average_ar}"
    )

import os
from dotenv import load_dotenv
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_relevancy,
)
from ragas import evaluate
from datasets import Dataset
import pandas as pd

from tests.test_utils import append_to_file, extract_eval_data
import logging

# Set the logging level for `httpx` to WARNING or higher
logging.getLogger("httpx").setLevel(logging.WARNING)

"""
Faithfulness - Measures the factual consistency of the answer to the context based on the question.
Context_relevancy - Measures how relevant the retrieved context is to the question, conveying the quality of the retrieval pipeline.
Answer_relevancy - Measures how relevant the output is to the query.
"""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAGAS_DO_NOT_TRACK = os.getenv("RAGAS_DO_NOT_TRACK")

def compute_ragas_results(dataset: Dataset, get_context_relevance:bool=True, verbose: bool=False) -> pd.DataFrame:
    if get_context_relevance:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_relevancy
            ],
        )
    else:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy
            ],
        )

    if verbose:
        print(result.scores[0]["faithfulness"])
        print(result.scores[0]["answer_relevancy"])
        print(result.scores[0]["context_relevancy"])

    df = result.to_pandas()
    return df

def compute_reverse_context_relevancies(dataset: Dataset, verbose: bool=False) -> pd.DataFrame:
    """
    Note that this one takes question and context as input from the dataset, except now the question is the generated exercise instead of the user query.
    """
    result = evaluate(
        dataset,
        metrics=[
            context_relevancy
        ],
    )

    df = result.to_pandas()

    new_column_names = {
        'context_relevancy': 'reverse_context_relevancy',
        'question': 'generated question'
    }

    # Rename the columns
    df = df.rename(columns=new_column_names)
    return df
    
if __name__ == "__main__":
    # DATA_DIR = os.getenv("DATA_DIR_PATH")

    # data_file = os.path.join(DATA_DIR, "results", "7-pipeline-llama3.json")
    # results_file = os.path.join(DATA_DIR, "results", "results.txt")
    # csv_file_ragas = os.path.join(DATA_DIR, "results", "pipeline-7-ragas-analysis.csv")

    # pipeline_data = extract_eval_data(data_file)[280:300]

    # data = {
    #     'question': [item.query.query for item in pipeline_data],
    #     'answer': [item.response.text for item in pipeline_data],
    #     'contexts' : [[doc.source.chunk for doc in item.retrieved_docs] for item in pipeline_data],
    # }

    # dataset = Dataset.from_dict(data)

    # df_ragas = compute_ragas_results(dataset, get_context_relevance=False)

    # data_reverse = {
    #     'question': [item.response.text for item in pipeline_data],
    #     'contexts' : [[doc.source.chunk for doc in item.retrieved_docs] for item in pipeline_data],
    # }

    # dataset_reverse = Dataset.from_dict(data_reverse)

    # df_rcr = compute_reverse_context_relevancies(dataset_reverse)

    # # Select only the desired column ('reverse_context_relevancy') from df_rcr
    # df_rcr = df_rcr[['reverse_context_relevancy']]

    # """
    # Here I merge the two dataframes
    # """
    # # Make sure that both DataFrames have the same number of rows
    # assert len(df_ragas) == len(df_rcr), "DataFrames must have the same length"

    # # Reset indices if needed to ensure alignment by index
    # df_ragas.reset_index(drop=True, inplace=True)
    # df_rcr.reset_index(drop=True, inplace=True)

    # # Concatenate along columns
    # merged_df = pd.concat([df_ragas, df_rcr], axis=1)

    # """
    # Here I store the results in a CSV
    # """
    # # Append the DataFrame to the existing CSV file
    # merged_df.to_csv(csv_file_ragas, mode='a', index=False, header=False) # set index=False

    """
    Here I compute the metric averages and store them in the results file
    """
    DATA_DIR = os.getenv("DATA_DIR_PATH")
    results_file = os.path.join(DATA_DIR, "results", "results.txt")
    df = pd.read_csv(os.path.join(DATA_DIR, "results", "pipeline-7-ragas-analysis.csv"))

    # Columns to calculate the mean
    columns_to_average = ['faithfulness', 'answer_relevancy', 'reverse_context_relevancy']  # Specify the columns you want to average

    """COMPUTE THIS FROM THE CSV FILE INSTEAD"""
    # Replace NaN with 0 and compute the mean for specified columns
    mean_values = df[columns_to_average].fillna(0).mean()

    # Accessing specific averages from the result
    average_gr = mean_values['faithfulness']
    average_ar = mean_values['answer_relevancy']
    # average_cr = mean_values['context_relevancy']
    average_rcr = mean_values['reverse_context_relevancy']
    
    append_to_file(results_file, f"Pipeline (7) RAGAS Groundedness: {average_gr}, Answer Relevancy: {average_ar}, Reverse Context Relevancy: {average_rcr}")
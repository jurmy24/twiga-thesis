
import openai
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

"""
This is the modules file, which will contain the modular components that can be used by the RAG pipelines I build.
"""

# TODO: Make it possible for the query_rewriter to see the entire conversation history so that it can add meat to the bone's of a message like "Write a question about what I said in my last message."

def query_rewriter(query, api_key):
    """
    Enhances a user query by rewriting it in better English and adding more detail.

    Parameters:
    - query (str): The user's original query string.
    - api_key (str): The API key for OpenAI.

    Returns:
    A string containing the enhanced query.
    """
    openai.api_key = api_key

    try:
        # Construct a prompt to both improve the query's English and add more detail
        prompt = (f"Rewrite the following query in better English, making it clearer "
                  f"and more detailed for retrieval in a vector database:\n\n{query}")

        # Send the prompt to the OpenAI API using a suitable engine
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can choose the model based on your needs
            prompt=prompt,
            temperature=0.5,  # Adjust for creativity. Lower values make the response more deterministic.
            max_tokens=100,  # Adjust based on the expected length of the enhanced query
            n=1,  # Number of completions to generate
            stop=None  # Optional stopping character or sequence
        )

        # Extract the enhanced query from the response
        enhanced_query = response.choices[0].text.strip()

        return enhanced_query
    except Exception as e:
        print(f"An error occurred: {e}")
        return query  # Return the original query in case of an error
    

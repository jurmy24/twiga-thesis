import openai
from dotenv import load_dotenv
import os

"""
This is the most basic implementation of the tool and I might rewrite it as a simple function instead.
"""

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class QuestionGenerator:
    def __init__(self):
        """
        Initializes the question generator with the necessary API key.

        Parameters:
        - api_key (str): The API key for OpenAI.
        """
        openai.api_key = OPENAI_API_KEY

    def generate_questions(self, topic, n=5):
        """
        Generates a list of questions based on the provided topic.

        Parameters:
        - topic (str): The topic to generate questions about.
        - n (int): The number of questions to generate.

        Returns:
        A list of strings, where each string is a question.
        """
        try:
            prompt = f"Generate {n} insightful questions about the topic: {topic}"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=100,
                n=1,
                stop=None
            )
            questions = response.choices[0].text.strip().split('\n')
            return questions
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

# Example usage
if __name__ == "__main__":
    topic = "Climate Change"
    question_generator = QuestionGenerator()
    questions = question_generator.generate_questions(topic, n=5)
    for question in questions:
        print(question)

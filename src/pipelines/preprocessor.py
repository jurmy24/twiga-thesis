import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def local_preprocessor(query:str):
    assert torch.cuda.is_available()

    torch.set_default_device("cuda")

    # Retrieve the microsoft phi-1.5 model (1.5 billion parameters)
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

    inputs = tokenizer(f"Write a passage that answers the given query: \nQuery: {query} \nAnswer: ", return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=100)
    text = tokenizer.batch_decode(outputs)[0]
    text = text.replace("Write a passage that answers the given query: \nQuery: {query} \nAnswer: ", "")
    return text.replace("\n", "") # because this model has a tendency to add unnecessary stuff if it tries to write another paragraph

def preprocessor(query: str):

    # Make a call to the "davinci" model
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that rewrites queries from the user into a short passage about the topic they are requesting a question about. You do not write a question, but find the topic they are requesting a question about and describe that topic."},
            {"role": "user", "content": f"{query}"}
        ]
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    
    res = preprocessor("give me defining question on chimpanzees for my form 2 students")
    print(res)
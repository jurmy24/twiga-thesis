import os
from dotenv import load_dotenv

from src.prompt_templates import ASSISTANT_OPENAI_PROMPT

import openai

"""
Assistant:	Purpose-built AI that uses OpenAI’s models and calls tools
Thread:	    A conversation session between an Assistant and a user. Threads store Messages and automatically handle truncation to fit content into a model’s context.
Message:	A message created by an Assistant or a user. Messages can include text, images, and other files. Messages stored as a list on the Thread.
Run:	    An invocation of an Assistant on a Thread. The Assistant uses its configuration and the Thread’s Messages to perform tasks by calling models and tools. As part of a Run, the Assistant appends Messages to the Thread.
Run Step:	A detailed list of steps the Assistant took as part of a Run. An Assistant can call tools or create Messages during its run. 
            Examining Run Steps allows you to introspect how the Assistant is getting to its final results.
"""

"""
The file_search tool implements several retrieval best practices out of the box to help you extract the right data from your files and augment the model’s responses. The file_search tool:

Rewrites user queries to optimize them for search.
Breaks down complex user queries into multiple searches it can run in parallel.
Runs both keyword and semantic searches across both assistant and thread vector stores.
Reranks search results to pick the most relevant ones before generating the final response.

By default, the file_search tool uses the following settings:

Chunk size: 800 tokens
Chunk overlap: 400 tokens
Embedding model: text-embedding-3-large at 256 dimensions
Maximum number of chunks added to context: 20 (could be fewer)
"""

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG = os.getenv("OPENAI_ORG")
TWIGA_ASSISTANT_ID = os.getenv("TWIGA_OPENAI_ASSISTANT_ID")

client = openai.OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG)

def create_assistant():
    # Create the assistant (choice of model: "gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09")
    assistant = client.beta.assistants.create(
    name="Tanzania Secondary School Exercise/Question Generator",
    instructions=ASSISTANT_OPENAI_PROMPT,
    model="gpt-3.5-turbo-0125",
    tools=[{"type": "file_search"}],
    )

    # Create a vector store
    vector_store = client.beta.vector_stores.create(name="Twiga Documents")
    
    # Ready the files for upload to OpenAI 
    file_path = os.path.join(DATA_DIR, "documents", "pdf", "GEOGRAPHY-F2-TIE.pdf")
    file_streams = open(file_path, "rb")
    
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,and poll the status of the file batch for completion.
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
    )
    
    # You can print the status and the file counts of the batch to see the result of this operation. 
    print(file_batch.status)
    print(file_batch.file_counts)

create_assistant()

# Vector Store ID (Twiga Documents) = vs_n0Skz00QPS3MFLXyEfWuKjaH

# # Retrieve the Assistant
# assistant = client.beta.assistants.retrieve(TWIGA_ASSISTANT_ID)

# assistant = client.beta.assistants.update(
#   assistant_id=assistant.id,
#   tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
# )

# # Upload the user provided file to OpenAI
# message_file = client.files.create(
#   file=open("edgar/aapl-10k.pdf", "rb"), purpose="assistants"
# )
 
# # Create a thread and attach the file to the message
# thread = client.beta.threads.create(
#   messages=[
#     {
#       "role": "user",
#       "content": "How many shares of AAPL were outstanding at the end of of October 2023?",
#       # Attach the new file to the message.
#       "attachments": [
#         { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
#       ],
#     }
#   ]
# )
 
# # The thread now has a vector store with that file in its tool resources.
# print(thread.tool_resources.file_search)


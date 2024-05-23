import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from app.utils.openai_utils import (
    check_if_thread_exists,
    print_conversation,
    store_thread,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID = os.getenv("TWIGA_OPENAI_ASSISTANT_ID")
OPENAI_ORG = os.getenv("OPENAI_ORG")
client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG)

logger = logging.getLogger(__name__)


def run_assistant(thread, name):
    # Retrieve the Assistant
    assistant = client.beta.assistants.retrieve(OPENAI_ASSISTANT_ID)
    # logging.info(f"This is the assistant's instructions: {assistant.instructions}")

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions=f"You are having a conversation with Tim. The user is a Geography teacher.",
    )

    # Wait for completion
    # https://platform.openai.com/docs/assistants/how-it-works/runs-and-run-steps#:~:text=under%20failed_at.-,Polling%20for%20updates,-In%20order%20to
    while run.status != "completed":
        # Be nice to the API
        time.sleep(0.5)
        logger.info(f"🏃‍♂️ Run status: {run.status}")
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    else:
        logger.info(f"🏁 Run completed")
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        # Print the conversation
        print("MESSAGE HISTORY")
        print("-----------------------------------------------------------")
        print_conversation(messages)
        print("-----------------------------------------------------------")
        new_message = messages.data[0].content[0].text.value

        logger.info(f"Generated message: {new_message}")

        return new_message


def generate_response(message_body, wa_id, name):
    # Check if there is already a thread_id for the wa_id
    thread_id = check_if_thread_exists(wa_id)
    # logger.info(f"This is the thread ID:{thread_id}")

    # If a thread doesn't exist, create one and store it
    if thread_id is None:
        logger.info(f"Creating new thread for {name} with wa_id {wa_id}")
        thread = client.beta.threads.create()
        store_thread(wa_id, thread.id)
        thread_id = thread.id
    else:  # Otherwise, retrieve the existing thread
        logger.info(f"Retrieving existing thread for {name} with wa_id {wa_id}")
        thread = client.beta.threads.retrieve(thread_id)

    # Add message to thread
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_body,
    )

    # Run the assistant and get the new message
    new_message = run_assistant(thread, name)

    return new_message

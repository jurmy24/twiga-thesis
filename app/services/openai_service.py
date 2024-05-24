import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.beta import Thread

from app.tools.generate_exercise import process_query
from app.utils.database_utils import check_if_thread_exists, store_thread
from app.utils.openai_utils import print_conversation

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID = os.getenv("TWIGA_OPENAI_ASSISTANT_ID")
OPENAI_ORG = os.getenv("OPENAI_ORG")
client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG)

logger = logging.getLogger(__name__)


def run_assistant(thread: Thread, message: str, verbose: bool = False):
    # Retrieve the Assistant
    assistant = client.beta.assistants.retrieve(OPENAI_ASSISTANT_ID)

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # Wait for completion
    while run.status != "completed":
        time.sleep(0.5)  # Be nice to the API
        logger.info(f"🏃‍♂️ Run status: {run.status}")
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        if run.status == "requires_action":
            logger.info("🔧 Action required")

            # Define the list to store tool outputs
            tool_outputs = []

            for tool in run.required_action.submit_tool_outputs.tool_calls:
                if tool.function.name == "generate_exercise":
                    tool_outputs.append(
                        {"tool_call_id": tool.id, "output": process_query(message)}
                    )

            if tool_outputs:
                # Send the response back to the function calling tool
                run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=run.thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )

        # RUN STATUS: EXPIRED | FAILED | CANCELLED | INCOMPLETE
        if run.status in ["expired", "failed", "cancelled", "incomplete"]:
            return {"response": run.last_error, "status_code": 500}

    logger.info(f"🏁 Run completed")
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    new_message = messages.data[0].content[0].text.value

    if verbose:
        print("MESSAGE HISTORY")
        print("-----------------------------------------------------------")
        print_conversation(messages)
        print("-----------------------------------------------------------")

    return new_message


def generate_response(message_body, wa_id, name):
    # Check if there is already a thread_id for the wa_id
    thread_id = check_if_thread_exists(wa_id)

    # If a thread doesn't exist, create one and store it
    if thread_id is None:
        logger.info(f"Creating new thread for {name} with wa_id {wa_id}")

        thread = client.beta.threads.create()
        store_thread(wa_id, thread.id)
        thread_id = thread.id
    else:  # Otherwise, retrieve the existing thread
        logger.info(f"Retrieving existing thread for {name} with wa_id {wa_id}")
        thread = client.beta.threads.retrieve(thread_id)

    # Add message to the relevant assistant thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_body,
    )

    # Run the assistant and get the new message
    new_message = run_assistant(thread, message_body, verbose=True)

    return new_message

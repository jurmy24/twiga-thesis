import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from app.utils.database_utils import store_thread

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
client = OpenAI(api_key=OPENAI_API_KEY)


def create_thread(wa_id: str, intro_msg: str):

    thread = client.beta.threads.create()
    store_thread(wa_id, thread.id)

    # Add message to thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="assistant",
        content=intro_msg,
    )
    return thread


# Define the Message and ContentBlock classes as per your data structure
class Text:
    def __init__(self, annotations: List, value: str):
        self.annotations = annotations
        self.value = value


class TextContentBlock:
    def __init__(self, text: Text, type: str):
        self.text = text
        self.type = type


class Message:
    def __init__(
        self,
        id: str,
        assistant_id: str,
        attachments: List,
        completed_at: int,
        content: List[TextContentBlock],
        created_at: int,
        incomplete_at: int,
        incomplete_details: dict,
        metadata: dict,
        object: str,
        role: str,
        run_id: str,
        status: str,
        thread_id: str,
    ):
        self.id = id
        self.assistant_id = assistant_id
        self.attachments = attachments
        self.completed_at = completed_at
        self.content = content
        self.created_at = created_at
        self.incomplete_at = incomplete_at
        self.incomplete_details = incomplete_details
        self.metadata = metadata
        self.object = object
        self.role = role
        self.run_id = run_id
        self.status = status
        self.thread_id = thread_id


class SyncCursorPage:
    def __init__(
        self,
        data: List[Message],
        object: str,
        first_id: str,
        last_id: str,
        has_more: bool,
    ):
        self.data = data
        self.object = object
        self.first_id = first_id
        self.last_id = last_id
        self.has_more = has_more


def print_conversation(messages: SyncCursorPage):
    if messages is not None:
        for message in reversed(list(messages.data)):
            if message.role in ["user", "assistant"]:
                sender = "User" if message.role == "user" else "Assistant"
                content = " ".join(
                    block.text.value
                    for block in message.content
                    if block.type == "text"
                )
                print(f"### {sender}: {content}")

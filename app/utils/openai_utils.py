import logging
import os
import shelve
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
client = OpenAI(api_key=OPENAI_API_KEY)


def clear_threads_db():
    with shelve.open("threads_db", writeback=True) as threads_shelf:
        threads_shelf.clear()


def upload_file(path):
    # Upload a file with an "assistants" purpose
    file = client.files.create(
        file=open("../../data/airbnb-faq.pdf", "rb"), purpose="assistants"
    )


def create_assistant(file):
    """
    You currently cannot set the temperature for Assistant via the API.
    """
    assistant = client.beta.assistants.create(
        name="Twiga Bot",
        instructions="You are Twiga, a WhatsApp bot designed by the Tanzania AI Community for secondary school teachers in Tanzania. You are kind and fulfill the requests of your users directly. ",
        model="gpt-4o",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "generate_exercise",
                    "description": "Generate an exercise or question for the students based on course literature",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_query": {
                                "type": "string",
                                "description": "A short message describing the desired question/exercise type and the topic it should be about",
                            }
                        },
                        "required": ["user_query"],
                    },
                },
            },
        ],
    )
    return assistant


def inspect_threads_db():
    with shelve.open("threads_db") as threads_shelf:
        if len(threads_shelf) == 0:
            print("The threads database is empty.")
        else:
            for key in threads_shelf:
                print(f"Key: {key} -> Value: {threads_shelf[key]}")


from typing import List


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


if __name__ == "__main__":
    # Clear the database
    clear_threads_db()
    print("Threads database cleared.")

    # Inspect the database
    inspect_threads_db()

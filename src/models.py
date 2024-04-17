from typing import Literal
from pydantic import BaseModel

# Define a ChatMessage type
class ChatMessage(BaseModel):
    content: str
    role: Literal["system", "user", "assistant"]

# Define the format for the chunk data
class Document(BaseModel):
    title: str
    contents: str
    created_on: str

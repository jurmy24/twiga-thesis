from typing import Literal, List, Optional
from pydantic import BaseModel

# Define a ChatMessage type
class ChatMessage(BaseModel):
    content: str
    role: Literal["system", "user", "assistant"]

# Define the format for the chunk data
class DocumentSchema(BaseModel):
    title: str
    contents: str
    created_on: str

class IndexSchema(BaseModel):
    content: str
    summary: str
    name: str
    url: str
    created_on: str
    updated_at: str
    category: str
    rolePermissions: List[str]

class Metadata(BaseModel):
    header_1: Optional[str] = None
    header_2: Optional[str] = None
    header_3: Optional[str] = None

class ChunkSchema(BaseModel):
    content: str
    metadata: Metadata




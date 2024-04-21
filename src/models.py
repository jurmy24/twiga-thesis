from typing import Literal, List, Optional
from pydantic import BaseModel

# Define a ChatMessage type
class ChatMessage(BaseModel):
    content: str
    role: Literal["system", "user", "assistant"]

class Metadata(BaseModel):
    title: Optional[str] = None
    chapter: Optional[str] = None
    subsection: Optional[str] = None

class ChunkSchema(BaseModel):
    page_content: str
    metadata: Metadata
    type: str # this is just "Document" for all the entries...


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
    subsubsection: Optional[str] = None
    type: Literal["Content", "Exercise"]
    exercise_format: Optional[Literal['short-answer', 'long-answer', 'true-false', "multiple-choice", "match", "draw"]] = None

class ChunkSchema(BaseModel):
    chunk: str # this is the stuff that shall be embedded
    metadata: Metadata

from typing import Literal, List, Optional
from pydantic import Field, BaseModel

# Define a ChatMessage type
class ChatMessage(BaseModel):
    content: str
    role: Literal["system", "user", "assistant"]

# class EvalQuery(BaseModel):
#     query: str
#     requested_exercise_format: Literal['short-answer', 'long-answer', 'true-false']
#     topic: str

class EvalQuery(BaseModel):
    query: str = Field(..., description="The query provided by the Tanzanian teacher asking the model to generate an exercise/question.")
    requested_exercise_format: Literal['short-answer', 'long-answer', 'true-false'] = Field(..., description="The type of question or exercise that is being requested.")
    topic: str

class EvalQueryList(BaseModel):
    queries: List[EvalQuery] = Field(..., description="A list of queries from Tanzanian teachers")

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

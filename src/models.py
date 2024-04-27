from typing import Literal, List, Optional
from pydantic import Field, BaseModel

# Define a ChatMessage type
class ChatMessage(BaseModel):
    content: str
    role: Literal["system", "user", "assistant"]

class RewrittenQuery(BaseModel):
    rewritten_query_str: str
    embedding: List[float]

class EvalQuery(BaseModel):
    query: str = Field(..., description="The query provided by the Tanzanian teacher asking the model to generate an exercise/question.")
    requested_exercise_format: Literal['short-answer', 'long-answer', 'true-false'] = Field(..., description="The type of question or exercise that is being requested.")
    topic: str
    embedding: Optional[List[float]]
    rewritten_query: Optional[RewrittenQuery]

    def to_dict(self):
        if self.embedding is not None and self.rewritten_query is not None:
            return {
                "query": self.query,
                "requested_exercise_format": self.requested_exercise_format,
                "topic": self.topic,
                "embedding": self.embedding,
                "rewritten_query": {
                    "rewritten_query_str": self.rewritten_query.rewritten_query_str,
                    "embedding": self.rewritten_query.embedding
                }
            }
        else:
            return {
                "query": self.query,
                "requested_exercise_format": self.requested_exercise_format,
                "topic": self.topic
            }

class Metadata(BaseModel):
    title: Optional[str] = None
    chapter: Optional[str] = None
    subsection: Optional[str] = None
    subsubsection: Optional[str] = None
    doc_type: Literal["Content", "Exercise"]
    exercise_format: Optional[Literal['short-answer', 'long-answer', 'true-false', "multiple-choice", "match", "draw"]] = None

class ChunkSchema(BaseModel):
    chunk: str # this is the stuff that shall be embedded
    metadata: Metadata
    embedding: List[float]

class RetrievedDocSchema(BaseModel):
    retrieval_type: str
    score: Optional[float]
    rank: Optional[int]
    id: str
    source: ChunkSchema

class ResponseSchema(BaseModel):
    text: str
    embedding: List[float]

class PipelineData(BaseModel):
    query: EvalQuery # this contains the query string, the requested exercise format, the topic, the embedding, and the string and embedding of the rewritten query for retrieval
    retrieved_docs: List[RetrievedDocSchema]
    response: ResponseSchema
    


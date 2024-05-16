from typing import Literal
from pydantic import BaseModel

class QuestionResSchema(BaseModel):
    answer_relevance: int
    formulation: int
    suitability: int

class MDGScore(BaseModel):
    query: str
    response: str
    exercise_format: Literal["long-answer", "short-answer", "true-false"]
    respondent_id: str
    result: QuestionResSchema
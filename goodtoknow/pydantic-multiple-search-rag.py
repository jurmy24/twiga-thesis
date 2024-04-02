import asyncio
import datetime
import enum
import instructor
from pydantic import BaseModel, ValidationError, BeforeValidator
from typing_extensions import Annotated
from openai import OpenAI
from instructor import llm_validator
from typing import Optional, List
import openai

# This could be very useful for RAG
class SearchType(str, enum.Enum):
    VIDEO = "video"
    EMAIL = "email"

class Search(BaseModel):
    title: str
    query: str
    before_date: datetime
    type: SearchType

    async def execute(self):
        pass

class MultipleSearch(BaseModel):
    searches: List[Search]

    async def execute(self):
        return await asyncio.gather(*[s.execute() for s in self.searches])
    
def segment(data: str) -> MultipleSearch:
    return openai.ChatCompletion.create(
        responseModel=MultipleSearch,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant",
            },
            {
                "role": "user",
                "content": f"Consider the data below:\n{data} and segment it into multiple search queries",
            }
        ],
        max_tokens=1000
    )


queries = segment("Please send me the video from last week about the investment case study and also documents about your GDPR policy.")

queries.execute()
# It searches for "Video" with the query "investment case study" using SearchType.VIDEO
# It searches for "Documents" with the query "GPDR policy" using SearchType.EMAIL at the same time
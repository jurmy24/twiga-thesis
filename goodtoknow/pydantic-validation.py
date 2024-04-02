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

# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# Define your desired output structure
class UserInfo(BaseModel):
    userMessage: Optional[str]
    name: str
    age: Annotated[int, BeforeValidator(llm_validator("This must be a reasonable number for an age", client=client)),] # ex. validation

        
try:
    # Extract structured data from natural language
    user_info = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserInfo,
        max_retries=2,
        messages=[{"role": "user", "content": "Clarissa is a man."}],
    )

    print(user_info.name)  # "John Doe"
    #> John Doe
    print(user_info.age)  # 30
    #> 30

except ValidationError as e:
    print(e)
    """
    1 validation error for QuestionAnswer
    answer
      Assertion failed, The age is not reasonable. [type=assertion_error, input_value='The meaning of life is to be evil and steal', input_type=str]
        For further information visit https://errors.pydantic.dev/2.6/v/assertion_error
    """




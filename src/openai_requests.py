import backoff
import json
import os
import openai
from src.utils import num_tokens_from_string
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG = os.getenv("OPENAI_ORG")

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=10, max_time=300)
async def async_openai_request() -> ChatCompletion:
    try:
        client = openai.AsyncOpenAI(
            api_key=OPENAI_API_KEY, organization=OPENAI_ORG
        )
    except openai.RateLimitError as e:
        raise
    except Exception as e:
        raise

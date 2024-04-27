from typing import Literal
import backoff
import json
import os

import groq
from groq import Groq
from groq import AsyncGroq
from src.utils import num_tokens_from_string

from dotenv import load_dotenv
import logging


# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

sync_client = Groq(api_key=GROQ_API_KEY)
async_client = AsyncGroq(api_key=GROQ_API_KEY)

# TODO: include the number of tokens in the verbose

# Decorator to automatically back off and retry on rate limit errors (set 60 second interval since Groq refreshes its rates every minute)
@backoff.on_exception(backoff.expo, groq.RateLimitError, interval=60, max_tries=10, max_time=300)
def groq_request(llm: Literal["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], verbose: bool=False, **params):
    try:
       # Print messages if the flag is True
        if verbose:
            logger.info(
                "Messages sent to API: ",
                json.dumps(params["messages"], indent=2),
            )

        completion = sync_client.chat.completions.create(
            model=llm
            **params
        )

        return completion
    except groq.RateLimitError as e:
        # Log and re-raise rate limit errors
        logger.error(f"Rate limit error: {e}")
        raise
    except Exception as e:
        # Log and re-raise unexpected errors
        logger.error(f"Unexpected error: {e}")
        raise

@backoff.on_exception(backoff.expo, groq.RateLimitError, interval=60, max_tries=10, max_time=300)
async def async_groq_request(llm: Literal["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], verbose: bool=False, **params):
    try:
        # Print messages if the flag is True
        if verbose:
            logger.info(
                "Messages sent to API: ",
                json.dumps(params["messages"], indent=2),
            )

        completion = await async_client.chat.completions.create(
            model=llm
            **params
        )

        return completion
    except groq.RateLimitError as e:
        # Log and re-raise rate limit errors
        logger.error(f"Rate limit error: {e}")
        raise
    except Exception as e:
        # Log and re-raise unexpected errors
        logger.error(f"Unexpected error: {e}")
        raise
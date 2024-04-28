from typing import Literal
import backoff
import json
import os
from dotenv import load_dotenv
import logging


import groq
from groq import Groq, AsyncGroq
from src.utils import num_tokens_from_messages

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING) # to get rid of the httpx logs

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

sync_client = Groq(api_key=GROQ_API_KEY)
async_client = AsyncGroq(api_key=GROQ_API_KEY)

# TODO: include the number of tokens in the verbose

# Decorator to automatically back off and retry on rate limit errors
@backoff.on_exception(backoff.expo, groq.RateLimitError, max_tries=10, max_time=300)
def groq_request(llm: Literal["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], verbose: bool=False, **params):
    try:
       # Print messages if the flag is True
        if verbose:
            messages = params.get('messages', None)
            logger.info(f"Messages sent to Groq API:\n{json.dumps(messages, indent=2)}")
            logger.info(f"Number of OpenAI-equivalent tokens in the payload:\n{num_tokens_from_messages(messages)}")
        
        full_params = {
            "model": llm,
            **params
        }

        completion = sync_client.chat.completions.create(
            **full_params
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

@backoff.on_exception(backoff.expo, groq.RateLimitError, max_tries=10, max_time=300)
async def async_groq_request(llm: Literal["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], verbose: bool=False, **params):
    try:
        # Print messages if the flag is True
        if verbose:
            messages = params.get('messages', None)
            logger.info(f"Messages sent to Groq API:\n{json.dumps(messages, indent=2)}")
            logger.info(f"Number of OpenAI-equivalent tokens in the payload:\n{num_tokens_from_messages(messages)}")

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
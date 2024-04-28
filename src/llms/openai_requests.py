from typing import Any
import backoff
import json
import os
import openai
from src.utils import num_tokens_from_messages
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG = os.getenv("OPENAI_ORG")

async_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG)
sync_client = openai.OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG)

# Decorator to automatically back off and retry on rate limit errors
@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=10, max_time=300)
def openai_request(verbose: bool=False, **params) -> ChatCompletion:
    try:
        # Print messages if the flag is True
        if verbose:
            messages = params.get('messages', None)
            logger.info(f"Messages sent to OpenAI API:\n{json.dumps(messages, indent=2)}")
            logger.info(f"Number of OpenAI-equivalent tokens in the payload:\n{num_tokens_from_messages(messages)}")

        completion = sync_client.chat.completions.create(
            **params
        )

        return completion
    except openai.RateLimitError as e:
        # Log and re-raise rate limit errors
        logger.error(f"Rate limit error: {e}")
        raise
    except Exception as e:
        # Log and re-raise unexpected errors
        logger.error(f"Unexpected error: {e}")
        raise

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=10, max_time=300)
async def async_openai_request(verbose: bool=False, **params) -> ChatCompletion:
    try:
        # Print messages if the flag is True
        if verbose:
            messages = params.get('messages', None)
            logger.info(f"Messages sent to OpenAI API:\n{json.dumps(messages, indent=2)}")
            logger.info(f"Number of OpenAI-equivalent tokens in the payload:\n{num_tokens_from_messages(messages)}")

        completion = await async_client.chat.completions.create(
            **params
        )

        return completion
    except openai.RateLimitError as e:
        raise
    except Exception as e:
        raise Exception(f"Failed to retrieve completion: {str(e)}")
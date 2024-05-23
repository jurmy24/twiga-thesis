import logging
import os
import shelve
from typing import List, Tuple

from app.utils.database_utils import get_user_state, update_user_state

# Initialize the logger for this module
logger = logging.getLogger(__name__)


def handle_onboarding(wa_id: str, message_body: str) -> Tuple[str, List[str] | None]:

    state = get_user_state(wa_id)
    user_state = state["state"]

    logger.info(f"This is the user state: {user_state}")

    if user_state == "start":
        update_user_state(wa_id, {"state": "ask_teacher"})
        return "Hello! I'm Twiga, your assistant. Are you a teacher?", ["Yes", "No"]

    elif user_state == "ask_teacher":
        if message_body.lower() == "yes":
            update_user_state(wa_id, {"state": "ask_subject"})
            return "What subject do you teach?", ["Math", "Physics", "Geography"]
        else:
            update_user_state(wa_id, {"state": "not_teacher"})
            return "This service is for teachers only.", None

    elif user_state == "ask_subject":
        subjects = ["Math", "Physics", "Geography"]
        logger.info(f"This is the message_body: {message_body}")
        if message_body in subjects:
            logger.info(f"This is the message_body: {message_body}")
            update_user_state(wa_id, {"state": "ask_form", "subject": message_body})
            return "Which form do you teach?", ["Form 1", "Form 2", "Form 3"]
        else:
            return "Please select a valid subject from the list.", subjects

    elif user_state == "ask_form":
        forms = ["Form 1", "Form 2", "Form 3"]
        if message_body in forms:
            subject = state["subject"]
            form = message_body
            update_user_state(wa_id, {"state": "completed", "form": form})
            welcome_message = (
                f"Welcome! You teach {subject} to {form}. How can I assist you today?"
            )
            return welcome_message, None
        else:
            return "Please select a valid form from the list.", forms

    elif user_state == "completed":
        # Onboarding complete, proceed with normal conversation
        return None

    # If we get here it crashes, allow it to solve itself
    return "I'm not sure how to proceed. Let's start over. Are you a teacher?", [
        "Yes",
        "No",
    ]

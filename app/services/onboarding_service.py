import logging
from typing import List, Optional, Tuple

from app.utils.database_utils import get_user_state, update_user_state

# Initialize the logger for this module
logger = logging.getLogger(__name__)


def handle_start(wa_id: str) -> Tuple[str, List[str]]:
    update_user_state(wa_id, {"state": "ask_teacher"})
    return (
        "Hello! My name is Twiga 🦒, I am a WhatsApp bot that supports teachers in the TIE curriculum with their daily tasks. \n\nAre you a TIE teacher?",
        ["Yes", "No"],
    )


def handle_ask_teacher(
    wa_id: str, message_body: str
) -> Tuple[str, Optional[List[str]]]:
    if message_body.lower() == "yes":
        update_user_state(wa_id, {"state": "ask_subject"})
        return "What subject do you teach?", ["Geography"]
    elif message_body.lower() == "no":
        update_user_state(wa_id, {"state": "not_teacher"})
        return "This service is for teachers only.", None
    else:
        return "Please select *Yes* or *No*. Are you a teacher?", ["Yes", "No"]


def handle_ask_subject(wa_id: str, message_body: str) -> Tuple[str, List[str]]:
    subjects = ["Math", "Physics", "Geography"]
    if message_body in subjects:
        update_user_state(wa_id, {"state": "ask_form", "subject": message_body})
        return "Which form do you teach?", ["Form 1", "Form 2", "Form 3", "Form 4"]
    else:
        return "Please select a valid subject from the list.", subjects


def handle_ask_form(
    wa_id: str, message_body: str, state: dict
) -> Tuple[str, List[str]]:
    forms = ["Form 1", "Form 2", "Form 3", "Form 4"]
    if message_body in forms:
        subject = state.get("subject")
        form = message_body
        update_user_state(wa_id, {"state": "completed", "form": form})
        welcome_message = (
            f"Welcome! You teach {subject} to {form}. How can I assist you today?"
        )
        return welcome_message, None
    else:
        return "Please select a valid form from the list.", forms


def handle_completed() -> Tuple[None, None]:
    # Onboarding complete, proceed with normal conversation
    return None, None


def handle_default() -> Tuple[str, List[str]]:
    return "I'm not sure how to proceed. Let's start over. Are you a teacher?", [
        "Yes",
        "No",
    ]


def handle_onboarding(wa_id: str, message_body: str) -> Tuple[str, Optional[List[str]]]:
    state = get_user_state(wa_id)
    user_state = state["state"]

    logger.info(f"This is the user state: {user_state}")

    if user_state == "start":
        return handle_start(wa_id)
    elif user_state == "ask_teacher":
        return handle_ask_teacher(wa_id, message_body)
    elif user_state == "ask_subject":
        return handle_ask_subject(wa_id, message_body)
    elif user_state == "ask_form":
        return handle_ask_form(wa_id, message_body, state)
    elif user_state == "completed":
        return handle_completed()
    else:
        return handle_default()

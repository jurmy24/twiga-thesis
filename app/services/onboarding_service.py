import logging
import os
import shelve
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()
SHELVE_FILENAME = os.getenv("SHELVE_FILENAME", "onboarding_db")


def clear_onboarding_db():
    with shelve.open(SHELVE_FILENAME, writeback=True) as db:
        db.clear()


def reset_conversation(wa_id):
    with shelve.open(SHELVE_FILENAME) as db:
        db[wa_id] = {"state": "start"}


def get_user_state(wa_id):
    with shelve.open(SHELVE_FILENAME) as db:
        return db.get(wa_id, {"state": "start"})


def update_user_state(wa_id, state_update):
    with shelve.open(SHELVE_FILENAME) as db:
        db[wa_id] = state_update


def handle_onboarding(wa_id, message_body) -> Tuple[str, List[str] | None]:
    state = get_user_state(wa_id)
    user_state = state["state"]

    logging.info(f"This is the user state: {user_state}")

    if user_state == "start":
        update_user_state(wa_id, {"state": "ask_teacher"})
        return "Hello! I'm Twiga, your assistant. Are you a teacher?", ["Yes", "No"]

    elif user_state == "ask_teacher":
        if message_body.lower() == "yes":
            update_user_state(wa_id, {"state": "ask_subject"})
            logging.info(f"USER STATE!!: {get_user_state(wa_id)}")
            return "What subject do you teach?", ["Math", "Physics", "Geography"]
        else:
            update_user_state(wa_id, {"state": "not_teacher"})
            return "This service is for teachers only.", None

    elif user_state == "ask_subject":
        subjects = ["Math", "Physics", "Geography"]
        if message_body in subjects:
            update_user_state(wa_id, {"state": "ask_form", "subject": message_body})
            return "Which form do you teach?", ["Form 1", "Form 2", "Form 3", "Form 4"]
        else:
            return "Please select a valid subject from the list.", subjects

    elif user_state == "ask_form":
        forms = ["Form 1", "Form 2", "Form 3", "Form 4"]
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

    return "I'm not sure how to proceed. Let's start over. Are you a teacher?", [
        "Yes",
        "No",
    ]


if __name__ == "__main__":
    # Clear the database
    clear_onboarding_db()
    print("Onboarding database cleared.")

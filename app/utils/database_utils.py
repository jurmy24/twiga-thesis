import os
import shelve
from typing import Dict


def clear_db(db_name: str):
    with shelve.open(db_name, writeback=True) as db:
        db.clear()


def inspect_db(db_name: str):
    with shelve.open(db_name) as db:
        if len(db) == 0:
            print(f"The {db_name} database is empty.")
        else:
            for key in db:
                print(f"Key: {key} -> Value: {db[key]}")


""" Users Database Functions """


def reset_conversation(wa_id: str, db_name: str = "users"):
    with shelve.open(db_name) as db:
        db[wa_id] = {"state": "start"}


def get_user_state(wa_id: str, db_name: str = "users"):
    with shelve.open(db_name) as db:
        return db.get(wa_id, {"state": "start"})


def update_user_state(wa_id: str, state_update: Dict[str, str], db_name: str = "users"):
    with shelve.open(db_name) as db:
        # Retrieve the existing state or create a new one if it doesn't exist
        existing_state = dict(db.get(wa_id, {}))

        # Update the existing state with the new state
        existing_state.update(state_update)

        # Save the updated state back to the database
        db[wa_id] = existing_state


""" Threads Database Functions """


def check_if_thread_exists(wa_id: str, db_name: str = "threads"):
    with shelve.open(db_name) as db:
        return db.get(wa_id, None)


def store_thread(wa_id: str, thread_id: str, db_name: str = "threads"):
    with shelve.open(db_name, writeback=True) as db:
        db[wa_id] = thread_id


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    USERS_DATABASE = os.getenv("USERS_DATABASE", "users")
    THREADS_DATABASE = os.getenv("THREADS_DATABASE", "threads")

    # Clear the threads database
    print("Threads database.")
    inspect_db(THREADS_DATABASE)
    clear_db(THREADS_DATABASE)
    print("Threads database cleared.")

    # # Clear the user-info database
    # print("Users database.")
    # inspect_db(USERS_DATABASE)
    # clear_db(USERS_DATABASE)
    # print("Onboarding database cleared.")

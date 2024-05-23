import os
import shelve

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


# Use context manager to ensure the shelf file is closed properly
def check_if_thread_exists(wa_id):
    with shelve.open("threads_db") as threads_shelf:
        return threads_shelf.get(wa_id, None)


def store_thread(wa_id, thread_id):
    with shelve.open("threads_db", writeback=True) as threads_shelf:
        threads_shelf[wa_id] = thread_id


def clear_threads_db():
    with shelve.open("threads_db", writeback=True) as threads_shelf:
        threads_shelf.clear()


def inspect_threads_db():
    with shelve.open("threads_db") as threads_shelf:
        if len(threads_shelf) == 0:
            print("The threads database is empty.")
        else:
            for key in threads_shelf:
                print(f"Key: {key} -> Value: {threads_shelf[key]}")


if __name__ == "__main__":
    # Clear the threads database
    clear_threads_db()
    print("Threads database cleared.")

    # Inspect the database
    inspect_threads_db()

    # Clear the user-info database
    clear_onboarding_db()
    print("Onboarding database cleared.")

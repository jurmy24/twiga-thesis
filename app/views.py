import asyncio
import datetime
import json
import logging
import threading
from typing import Tuple

from flask import Blueprint, Response, current_app, jsonify, request

from .decorators.security import signature_required
from .utils.database_utils import (
    get_message_count,
    increment_message_count,
    retrieve_messages,
    store_message,
)
from .utils.whatsapp_utils import (
    get_text_message_input,
    is_valid_whatsapp_message,
    process_text_for_whatsapp,
    process_whatsapp_message,
    send_message,
)

webhook_blueprint = Blueprint("webhooks", __name__)

logger = logging.getLogger(__name__)


# Initialize the in-memory dictionary to store message counts
message_counts = {}


# Function to reset counts at midnight (optional, if running a persistent service)
def reset_counts():
    now = datetime.datetime.now()
    midnight = datetime.datetime.combine(now.date(), datetime.time())
    seconds_until_midnight = (midnight + datetime.timedelta(days=1) - now).seconds
    threading.Timer(seconds_until_midnight, reset_counts).start()
    message_counts.clear()


def is_rate_limit_reached(wa_id: str) -> bool:

    count, last_message_time = get_message_count(wa_id)
    # Reset the message count if it's a new day
    if datetime.datetime.now().date() > last_message_time.date():
        count = 0

    # Check if the wa_id has sent more than 5 messages today
    daily_message_limit = 20
    if count >= daily_message_limit:
        return True

    # Increment the message count (this also handles new day resets)
    increment_message_count(wa_id)

    return False


async def handle_message() -> Tuple[Response, int]:
    """
    Handle incoming webhook events from the WhatsApp API.

    This function processes incoming WhatsApp messages and other events,
    such as delivery statuses. If the event is a valid message, it gets
    processed. If the incoming payload is not a recognized WhatsApp event,
    an error is returned.

    Every message send will trigger 4 HTTP requests to your webhook: message, sent, delivered, read.

    Returns:
        response: A tuple containing a JSON response and an HTTP status code.
    """
    body = request.get_json()

    # Check if it's a WhatsApp status update
    if (
        body.get("entry", [{}])[0]
        .get("changes", [{}])[0]
        .get("value", {})
        .get("statuses")
    ):
        logger.info("Received a WhatsApp status update.")
        return jsonify({"status": "ok"}), 200

    try:
        if is_valid_whatsapp_message(body):
            logger.info("Received a valid WhatsApp message.")

            message = body["entry"][0]["changes"][0]["value"]["messages"][0]
            message_timestamp = int(message.get("timestamp"))
            current_timestamp = int(datetime.datetime.now().timestamp())

            wa_id = body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]

            # Check if the message timestamp is within 10 seconds of the current time
            if current_timestamp - message_timestamp <= 10:
                # Check if the daily message limit has been reached
                if is_rate_limit_reached(wa_id):
                    logger.warning(f"Message limit reached for wa_id: {wa_id}")
                    sleepy_text = "🚫 You have reached your daily messaging limit, so Twiga 🦒 is quite sleepy 🥱 from all of today's texting . Let's talk more tomorrow!"
                    sleepy_msg = process_text_for_whatsapp(sleepy_text)
                    data = get_text_message_input(
                        current_app.config["RECIPIENT_WAID"],
                        sleepy_msg,  # could also just use wa_id here instead of going to config
                    )
                    store_message(wa_id, message, role="user")
                    store_message(
                        wa_id,
                        sleepy_text,
                        role="twiga",
                    )
                    await send_message(data)

                    return jsonify({"status": "ok"}), 200
                # This function is used to process and ultimately send a response message to the user
                await process_whatsapp_message(body)
                return jsonify({"status": "ok"}), 200
            else:
                store_message(wa_id, message, role="user")
                logger.warning("Received a message with an outdated timestamp.")
                return jsonify({"status": "ok"}), 200

        else:
            # if the request is not a WhatsApp API event, return an error
            return (
                jsonify({"status": "error", "message": "Not a WhatsApp API event"}),
                404,
            )
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON")
        return jsonify({"status": "error", "message": "Invalid JSON provided"}), 400


# Required webhook verifictaion for WhatsApp
def verify():
    # Parse params from the webhook verification request
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    # Check if a token and mode were sent
    if mode and token:
        # Check the mode and token sent are correct
        if mode == "subscribe" and token == current_app.config["VERIFY_TOKEN"]:
            # Respond with 200 OK and challenge token from the request
            logger.info("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            # Responds with '403 Forbidden' if verify tokens do not match
            logger.error("VERIFICATION_FAILED")
            return jsonify({"status": "error", "message": "Verification failed"}), 403
    else:
        # Responds with '400 Bad Request'
        logger.error("MISSING_PARAMETER")
        return jsonify({"status": "error", "message": "Missing parameters"}), 400


@webhook_blueprint.route("/webhooks", methods=["GET"])
def webhook_get():
    return verify()


@webhook_blueprint.route("/webhooks", methods=["POST"])
@signature_required
def webhook_post():
    return asyncio.run(handle_message())

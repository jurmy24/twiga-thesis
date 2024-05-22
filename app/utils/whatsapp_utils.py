import asyncio
import json
import logging
import re

import aiohttp
import requests
from flask import current_app, jsonify

from app.services.onboarding_service import get_user_state, handle_onboarding
from app.services.openai_service import generate_response
from app.utils.openai_utils import check_if_thread_exists


def log_http_response(response):
    logging.info(f"Status: {response.status_code}")
    logging.info(f"Content-type: {response.headers.get('content-type')}")
    logging.info(f"Body: {response.text}")


def get_text_message_input(recipient, text):
    return json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
    )


def get_interactive_message_input(recipient, text, options):
    buttons = [
        {
            "type": "reply",
            "reply": {"id": f"option-{i}", "title": opt},
        }
        for i, opt in enumerate(options)
    ]

    return json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": text},
                "footer": {"text": "Twiga - your teaching assistant."},
                "action": {"buttons": buttons},
            },
        }
    )


def send_message(data):
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {current_app.config['ACCESS_TOKEN']}",
    }

    url = f"https://graph.facebook.com/{current_app.config['VERSION']}/{current_app.config['PHONE_NUMBER_ID']}/messages"

    try:
        response = requests.post(
            url, data=data, headers=headers, timeout=10
        )  # 10 seconds timeout as an example
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.Timeout:
        logging.error("Timeout occurred while sending message")
        return jsonify({"status": "error", "message": "Request timed out"}), 408
    except (
        requests.RequestException
    ) as e:  # This will catch any general request exception
        logging.error(f"Request failed due to: {e}")
        return jsonify({"status": "error", "message": "Failed to send message"}), 500
    else:
        # Process the response as normal
        log_http_response(response)
        return response


def process_text_for_whatsapp(text):
    # Remove brackets
    pattern = r"\【.*?\】"
    # Substitute the pattern with an empty string
    text = re.sub(pattern, "", text).strip()

    # Pattern to find double asterisks including the word(s) in between
    pattern = r"\*\*(.*?)\*\*"

    # Replacement pattern with single asterisks
    replacement = r"*\1*"

    # Substitute occurrences of the pattern with the replacement
    whatsapp_style_text = re.sub(pattern, replacement, text)

    return whatsapp_style_text


def process_whatsapp_message(body):
    wa_id = body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]
    name = body["entry"][0]["changes"][0]["value"]["contacts"][0]["profile"]["name"]

    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    message_type = message.get("type")

    if message_type == "text":
        message_body = message["text"]["body"]
    elif (
        message_type == "interactive"
        and message["interactive"]["type"] == "button_reply"
    ):
        message_body = message["interactive"]["button_reply"]["title"]
    else:
        logging.error(f"Unsupported message type: {message_type}")
        return

    # If the onboarding process is not completed, handle onboarding
    state = get_user_state(wa_id)
    if state["state"] != "completed":
        response, options = handle_onboarding(wa_id, message_body)
        response = process_text_for_whatsapp(response)
        if options:
            data = get_interactive_message_input(
                current_app.config["RECIPIENT_WAID"], response, options
            )
        else:
            data = get_text_message_input(
                current_app.config["RECIPIENT_WAID"], response
            )
    # Twiga Integration
    else:
        response = generate_response(message_body, wa_id, name)
        response = process_text_for_whatsapp(response)
        data = get_text_message_input(current_app.config["RECIPIENT_WAID"], response)

    send_message(data)


def is_valid_whatsapp_message(body):
    """
    Check if the incoming webhook event has a valid WhatsApp message structure.
    """
    return (
        body.get("object")
        and body.get("entry")
        and body["entry"][0].get("changes")
        and body["entry"][0]["changes"][0].get("value")
        and body["entry"][0]["changes"][0]["value"].get("messages")
        and body["entry"][0]["changes"][0]["value"]["messages"][0]
    )

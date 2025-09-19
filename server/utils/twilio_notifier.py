from twilio.rest import Client
from utils.config_loader import Config

# Create Twilio client once
twilio_client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

def send_whatsapp_message(image_url, message_text):
    """Send WhatsApp alert with image attached"""
    try:
        message = twilio_client.messages.create(
            body=message_text,
            from_="whatsapp:" + Config.TWILIO_PHONE,   # e.g., +14155238886
            to="whatsapp:" + Config.TO_PHONE,         # e.g., +91xxxxxx
            media_url=[image_url]                     # Cloudinary image
        )
        print(f"WhatsApp message sent: {message.sid}")
    except Exception as e:
        print(f"Twilio Error: {e}")

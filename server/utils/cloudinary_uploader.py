import cloudinary
import cloudinary.uploader
from utils.config_loader import Config

# Configure cloudinary
cloudinary.config(
    cloud_name=Config.CLOUDINARY_CLOUD_NAME,
    api_key=Config.CLOUDINARY_API_KEY,
    api_secret=Config.CLOUDINARY_API_SECRET
)

def upload_frame(frame_path):
    """Upload frame to Cloudinary and return URL"""
    try:
        response = cloudinary.uploader.upload(frame_path)
        return response["secure_url"]
    except Exception as e:
        print(f"Cloudinary Upload Error: {e}")
        return None

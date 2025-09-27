import cloudinary
import cloudinary.uploader
from utils.config_loader import Config

# Configure Cloudinary
cloudinary.config(
    cloud_name=Config.CLOUDINARY_CLOUD_NAME,
    api_key=Config.CLOUDINARY_API_KEY,
    api_secret=Config.CLOUDINARY_API_SECRET
)

def upload_frame(frame_path, folder_name):
   
    try:
        response = cloudinary.uploader.upload(
            frame_path,
            folder=folder_name  # Store in specified folder
        )
        return response.get("secure_url")
    except Exception as e:
        print(f"Cloudinary Upload Error: {e}")
        return None

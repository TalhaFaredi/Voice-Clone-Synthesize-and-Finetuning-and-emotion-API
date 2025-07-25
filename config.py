import os
import tempfile


UPLOAD_FOLDER1 = os.path.join(tempfile.gettempdir(), 'f5_uploads')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER1):
    os.makedirs(UPLOAD_FOLDER1, exist_ok=True)



APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

SPANISH_API_URL = "http://34.238.66.156:8000"
ENGLISH_API_URL = "http://34.238.66.156:5000"

def get_api_url(language):
    return ENGLISH_API_URL if language.lower() == 'english' else SPANISH_API_URL
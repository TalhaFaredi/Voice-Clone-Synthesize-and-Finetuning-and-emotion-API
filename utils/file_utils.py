# utils/file_utils.py

import os
import uuid
from config import UPLOAD_FOLDER
from werkzeug.utils import secure_filename
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'webm'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_audio_file(audio_file, voice_name):
    original_filename = audio_file.filename
    name_part, extension = os.path.splitext(original_filename)
    safe_voice_name = voice_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    unique_id = uuid.uuid4().hex
    filename = f"{safe_voice_name}_{unique_id}{extension}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(file_path)
    return os.path.join('uploads', filename)
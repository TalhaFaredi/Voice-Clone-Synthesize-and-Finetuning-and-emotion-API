# routes/finetune_routes.py

import logging
import os
from flask import Blueprint, request, jsonify,current_app
from services.db_service import get_db
from utils.file_utils import allowed_file, save_audio_file
from werkzeug.utils import secure_filename
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
finetune_routes = Blueprint('finetune_routes', __name__)

# Configuration

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size


# finetune_routes.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# finetune_routes.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


@finetune_routes.route('/submit_finetune', methods=['POST'])
def submit_finetune():
    language = request.form.get('language')
    project_name = request.form.get('project_name')
    audio_file = request.files.get('audio_file')

    if not language or not project_name or not audio_file:
        return jsonify({'error': 'Missing required fields'}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid audio file format'}), 400

    filename = secure_filename(f"{project_name}_{audio_file.filename}")
    project_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], project_name)
    os.makedirs(project_folder, exist_ok=True)

    file_path = os.path.join(project_folder, filename)
    audio_file.save(file_path)

    # Save to the database
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO finetuning_profiles (language, voice_name, audio_path)
                VALUES (%s, %s, %s)
            """, (language, project_name, file_path))
            conn.commit()

    return jsonify({'message': 'Fine-tuning data saved successfully!'}), 200

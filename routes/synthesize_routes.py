
import logging
import os
import time
import uuid
from flask import Blueprint, Response, request, jsonify,current_app
import psycopg2
import requests
from services.db_service import get_db
from utils.file_utils import allowed_file, save_audio_file
from datetime import datetime
from werkzeug.utils import secure_filename
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# SPANISH_API_URL = "http://34.238.66.156:5002"
# ENGLISH_API_URL = "http://34.238.66.156:5000" 
SPANISH_API_URL = "http://34.238.66.156:5000/es"
ENGLISH_API_URL = "http://34.238.66.156:5000/en"

def get_api_url(language ):
    """Return the appropriate API URL based on language selection"""
    if language and language.lower() == 'english':
        return ENGLISH_API_URL
    return SPANISH_API_URL  # Default to Spanish

synthesize_routes = Blueprint('synthesize_routes', __name__)

# Configuration

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

# synthesize_routes.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# synthesize_routes.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


@synthesize_routes.route('/api/synthesize', methods=['POST'])
def synthesize_text():
    try: 
        UPLOAD_FOLDER = os.path.join(current_app.root_path, 'static', 'uploads')
        language = request.form.get('language', '').lower() 
        print(language) # Default to Spanish
        project_name1 = request.form.get('project_name')
        project_name=str(project_name1).replace(f'{language}:', '')
        print(project_name)
     
        
        # Get the appropriate API URL based on language
        api_url = get_api_url(language)
        text = request.form.get('text')
        ref_audio = None
        
        if 'reference_audio' in request.files:
            ref_audio = request.files['reference_audio']
        
        if not project_name:
            return jsonify({"status": "error", "message": "Project name is required"}), 400
            
        if not text:
            return jsonify({"status": "error", "message": "Text to synthesize is required"}), 400
        
        # Create a multipart form-data request to the synthesize API
        payload = {'project_select': project_name, 'gen_text': text}
        
        # Always use model_last.pt as checkpoint
        payload['checkpoint'] = 'model_last.pt'
        
        # Handle reference audio if provided
        files = {}
        temp_path = None
        file_handle = None
        
        try:
            if ref_audio and ref_audio.filename:
                filename = secure_filename(ref_audio.filename)
                temp_path = os.path.join(UPLOAD_FOLDER, filename)
                ref_audio.save(temp_path)
                
                logger.info(f"Saved reference audio temporarily to {temp_path}")
                file_handle = open(temp_path, 'rb')
                
                content_type = ref_audio.content_type or 'audio/wav'
                files = {'ref_audio': (filename, file_handle, content_type)}
            
            # Call the synthesize API
            logger.info(f"Calling synthesize API for project: {project_name}")
            print(payload)
            print(api_url)
            response = requests.post(
                f"{api_url}/synthesize",
                data=payload,
                files=files
            )
            
            if response.status_code == 200:
                # Audio data is returned directly as binary content
                return Response(
                    response.content,
                    mimetype=response.headers.get('Content-Type', 'audio/wav'),
                    headers={'Content-Disposition': 'attachment; filename=synthesized_audio.wav'}
                )
            else:
                print(response.status_code)
                print(response.text)
                # Try to get error message from JSON if possible
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', f'{response.json()['error']}')
                except:
                    error_message = f"API returned status code {response.status_code}: {response.text[:200]}"
                
                logger.error(f"Synthesis error: {error_message}")
                return jsonify({
                    "status": "error",
                    "message": f"Failed to synthesize audio: {error_message}"
                }), 400
                
        finally:
            if file_handle:
                try:
                    file_handle.close()
                except:
                    pass
                    
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"Removed temporary file {temp_path}")
                except:
                    logger.warning(f"Failed to remove temporary file {temp_path}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API error during synthesis: {str(e)}")
        error_detail = ""
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                if isinstance(error_data, dict) and 'message' in error_data:
                    error_detail = f": {error_data['message']}"
            except:
                if e.response.text:
                    error_detail = f": {e.response.text[:200]}"
                    
        return jsonify({
            "status": "error", 
            "message": f"Failed during synthesis: {str(e)}{error_detail}"
        }), 500
    except Exception as e:
        logger.exception("Unexpected error during synthesis")
        return jsonify({
            "status": "error", 
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500


import logging
import os
import uuid
from flask import Blueprint, request, jsonify,current_app
import psycopg2
import requests
from services.db_service import get_db
from utils.file_utils import allowed_file, save_audio_file

from werkzeug.utils import secure_filename
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPANISH_API_URL = "http://34.238.66.156:5000/es"
ENGLISH_API_URL = "http://34.238.66.156:5000/en"


ALLOWED_EXTENSIONS = {'mp3', 'wav', 'webm'} 
def get_api_url(language ):
    """Return the appropriate API URL based on language selection"""
    if language and language.lower() == 'english':
        return ENGLISH_API_URL
    return SPANISH_API_URL  # Default to Spanish



project_routes = Blueprint('project_routes', __name__, static_folder='static')


ALLOWED_EXTENSIONS = {'mp3', 'wav', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size


# project_routes.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# project_routes.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

@project_routes.route('/api/get-projects', methods=['GET'])
def get_projects():
    try:
        language = request.args.get('language', 'english').lower()  # Default to Spanish
        
        # Get the appropriate API URL based on language
        api_url = get_api_url(language)
        
        # Call the external API to get projects
        logger.info(f"Fetching projects for language: {language} from API: {api_url}")
        response = requests.get(f"{api_url}/projects")
        response.raise_for_status()
        
        projects_data = response.json()
        projects = projects_data.get('projects', [])
        
        # Add language prefix to each project
        prefixed_projects = [f"{language}:{project}" for project in projects]
        
        return jsonify({
            "status": "success",
            "projects": prefixed_projects
        })
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API error fetching projects: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Failed to fetch projects: {str(e)}"
        }), 500
    except Exception as e:
        logger.exception("Unexpected error fetching projects")
        return jsonify({
            "status": "error", 
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500

@project_routes.route('/api/projects', methods=['POST'])
def create_project():
    try:
        project_name = request.form.get('project_name')
        language = request.form.get('language', 'spanish')  # Default to Spanish
        
        if not project_name:
            return jsonify({"status": "error", "message": "Project name is required"}), 400
            
        # Get the appropriate API URL based on language
        api_url = get_api_url(language)
        
        # Call the external API to create a project
        response = requests.post(
            f"{api_url}/projects", 
            json={"name": project_name, "tokenizer_type": "pinyin"}
        )
        
        response.raise_for_status()  # Raise an error for bad status codes
        result = response.json()
        
        return jsonify({
            "status": "success", 
            "message": f"Project '{project_name}' created successfully",
            "project_name": result.get('project_name', project_name)
        })
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API error creating project: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Failed to create project: {str(e)}"
        }), 500
    except Exception as e:
        logger.exception("Unexpected error creating project")
        return jsonify({
            "status": "error", 
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500

@project_routes.route('/api/upload', methods=['POST'])
def upload_audio():
    temp_path = None
    file_handle = None
    
    try:
        project_name = request.form.get('project_name')
        language = request.form.get('language', 'spanish')  # Default to Spanish
        
        if not project_name:
            return jsonify({"status": "error", "message": "Project name is required"}), 400
            
        # Check if file is present in the request
        if 'audio_file' not in request.files:
            return jsonify({"status": "error", "message": "No audio file provided"}), 400
            
        audio_file = request.files['audio_file']
        
        if audio_file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
            
        # if not allowed_file(audio_file.filename):
        #     return jsonify({
        #         "status": "error", 
        #         "message": f"File type not allowed. Please use: {', '.join(ALLOWED_EXTENSIONS)}"
        #     }), 400
            
        # Generate a filename if not provided
        original_filename = "upload.wav"
        if audio_file.filename:
            original_filename = audio_file.filename
            
        filename = secure_filename(original_filename)
        temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(temp_path)
        
        logger.info(f"Saved audio file temporarily to {temp_path}")
        
        # Prepare files for the transcribe API
        file_handle = open(temp_path, 'rb')
        
        # Set content type if not provided
        content_type = 'audio/wav'
        if audio_file.content_type:
            content_type = audio_file.content_type
            
        # Create files dictionary for requests
        files = {'voices': (filename, file_handle, content_type)}
        
        # Get language from project name if it has a prefix
        if ':' in project_name:
            language, actual_project_name = project_name.split(':', 1)
            project_name = actual_project_name
            
        # Get the appropriate API URL based on language
        api_url = get_api_url(language)
            
        # Call the transcribe API
        response = requests.post(
            f"{api_url}/transcribe/{project_name}", 
            files=files
        )
        
        # Check for API-specific errors even if status code is OK
        result = response.json()
        if response.status_code != 200 or (isinstance(result, dict) and result.get("status") == "error"):
            error_message = result.get("message", "Unknown API error") if isinstance(result, dict) else "Unknown API error"
            logger.error(f"API error in transcription response: {error_message}")
            
            # If error contains details from API, show them
            details = None
            if isinstance(result, dict) and "details" in result:
                details = result["details"]
                
            return jsonify({
                "status": "error", 
                "message": f"API Error: {error_message}",
                "details": details
            }), 400
        
        return jsonify({
            "status": "success", 
            "message": "Audio uploaded and transcribed successfully",
            "details": result
        })
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API error during transcription: {str(e)}")
        # Extract response content if available
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
            "message": f"Failed during transcription: {str(e)}{error_detail}"
        }), 500
    except Exception as e:
        logger.exception("Unexpected error during upload/transcription")
        return jsonify({
            "status": "error", 
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500
    finally:
        # Clean up resources
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

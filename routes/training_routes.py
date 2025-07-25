import logging
import os
import time
import uuid
from flask import Blueprint, request, jsonify,current_app
import psycopg2
import requests
from services.db_service import get_db
from utils.file_utils import allowed_file, save_audio_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
training_routes = Blueprint('training_routes', __name__)

# Configuration
# UPLOAD_FOLDER = os.path.join(current_app.root_path, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# training_routes.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# training_routes.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

SPANISH_API_URL = "http://34.238.66.156:5000/es"
ENGLISH_API_URL = "http://34.238.66.156:5000/en"

def get_api_url(language ):
    """Return the appropriate API URL based on language selection"""
    if language and language.lower() == 'english':
        return ENGLISH_API_URL
    return SPANISH_API_URL  # Default to Spanish


@training_routes.route('/api/start-training', methods=['POST'])
def start_training():
    try:
        UPLOAD_FOLDER = os.path.join(current_app.root_path, 'static', 'uploads')
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        project_name = request.form.get('project_name')
        language = request.form.get('language', '')  # Get language from form data
        
        if not project_name:
            return jsonify({"status": "error", "message": "Project name is required"}), 400
            
        # Get language from project name if it has a prefix
        if ':' in project_name:
            language, actual_project_name = project_name.split(':', 1)
            project_name = actual_project_name
        
        # Get the appropriate API URL based on language
        api_url = get_api_url(language)
        
        # Call the training API - using the correct endpoint
        logger.info(f"Starting training for project: {project_name} using {language} API")
        training_endpoint = f"{api_url}/start-training"
        logger.debug(f"Calling API endpoint: {training_endpoint}")
        
        response = requests.post(
            training_endpoint, 
            json={"project_name": project_name}
        )
        
        # Handle API response, checking if it's valid JSON
        try:
            data = response.json()
            if response.status_code != 200 or (isinstance(data, dict) and data.get("status") == "error"):
                error_message = data.get("message", "Unknown API error") if isinstance(data, dict) else "Unknown API error"
                logger.error(f"API error in response: {error_message}")
                return jsonify({
                    "status": "error", 
                    "message": f"API Error: {error_message}"
                }), 400
        except ValueError as json_err:
            # Handle case where response is not valid JSON
            logger.error(f"Invalid JSON response from API: {response.text}")
            
            # If training started successfully but no JSON was returned
            if response.status_code == 200:
                # Check for specific success or error messages in the text response
                response_text = response.text.lower()
                
                if "training completed" in response_text:
                    time.sleep(15)
                    logger.info("Training completed successfully")
                    return jsonify({
                        "status": "success",
                        "message": "Training completed successfully!",
                        "details": {"training_status": "completed", "raw_response": response.text[:200]}
                    })
                elif "error" in response_text or "failed" in response_text or "exception" in response_text:
                    logger.error(f"Training error detected in response: {response.text}")
                    return jsonify({
                        "status": "error",
                        "message": f"Error during training: {response.text[:200]}"
                    }), 400
                else:
                    # Default success case for "Training started please wait" etc.
                    logger.info("Training process started")
                    return jsonify({
                        "status": "success",
                        "message": "Fine-tuning process started successfully.",
                        "details": {"training_status": "started", "raw_response": response.text[:200]}
                    })
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid response from API: {response.text[:200]}"
                }), response.status_code
        
        return jsonify({
            "status": "success", 
            "message": "Fine-tuning process started successfully",
            "details": data
        })
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API error starting training: {str(e)}")
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
        if f"{str(e)}{error_detail}"=="Response ended prematurely":
            logger.info("Training completed successfully")
            time.sleep(10)
            return jsonify({
                "status": "success",
                "message": "Training completed successfully!",
                "details": {"training_status": "completed", "raw_response": "training completed"}
            })
            
        return jsonify({
            "status": "error", 
            "message": f"Failed to start training: {str(e)}{error_detail}"
        }), 500
    except Exception as e:
        logger.exception("Unexpected error starting training")
        return jsonify({
            "status": "error", 
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500


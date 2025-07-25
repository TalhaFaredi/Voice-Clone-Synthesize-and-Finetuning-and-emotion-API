import logging
import os
import time
from flask import Blueprint, request, jsonify,current_app
import requests
from werkzeug.utils import secure_filename
import pandas as pd
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
process_csv_routes = Blueprint('process_csv_routes', __name__)


SPANISH_API_URL = "http://34.238.66.156:5000/es"
ENGLISH_API_URL = "http://34.238.66.156:5000/en"
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'webm'} 
def get_api_url(language ):
    """Return the appropriate API URL based on language selection"""
    if language and language.lower() == 'english':
        return ENGLISH_API_URL
    return SPANISH_API_URL  # Default to Spanish


@process_csv_routes.route('/process-csv', methods=['POST'])
def process_csv():
    """Process CSV file and synthesize audio for each row sequentially on the server side."""
    try:
        # Check if language and project are provided
        language = request.form.get('language')
        project_name = request.form.get('project_name')
        
        if not language or not project_name:
            return jsonify({
                "status": "error",
                "message": "Language and project name are required"
            }), 400
        
        # Check if CSV file is provided
        if 'csv_file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "CSV file is required"
            }), 400
            
        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No CSV file selected"
            }), 400
            
        # Check if reference audio is provided - MANDATORY
        if 'reference_audio' not in request.files:
            return jsonify({
                "status": "error",
                "message": "Reference audio file is required"
            }), 400
            
        ref_audio = request.files['reference_audio']
        if ref_audio.filename == '':
            return jsonify({
                "status": "error",
                "message": "Reference audio file is required"
            }), 400
        
        # Save reference audio
        ref_filename = secure_filename(ref_audio.filename)
        ref_audio_path = os.path.join(current_app.config['UPLOAD_FOLDER'], ref_filename)
        ref_audio.save(ref_audio_path)
        logger.info(f"Reference audio saved to {ref_audio_path}")
        
        # Process CSV file
        try:
            # Read CSV file
            # csv_content = csv_file.read().decode('utf-8')
            csv_content = csv_file.read().decode('cp1252')
            csv_file.seek(0)  # Reset file pointer
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Check if CSV has at least 2 columns
            if len(df.columns) < 2:
                return jsonify({
                    "status": "error",
                    "message": "CSV must have at least 2 columns: filename and text"
                }), 400
            
            # Get the appropriate API URL based on language
            api_url = get_api_url(language)
            
            # Process each row sequentially and collect results
            synthesis_results = []
            
            for index, row in df.iterrows():
                filename = str(row.iloc[0])  # Using iloc to avoid deprecation warning
                text = str(row.iloc[1])
                
                # Create a result object
                result = {
                    "filename": filename,
                    "text": text,
                    "status": "pending"
                }
                
                # Create a multipart form-data request to the synthesize API
                payload = {
                    'project_select': project_name.replace(f'{language}:', ''),
                    'gen_text': text,
                    'checkpoint': 'model_last.pt'
                }
                
                # Handle reference audio - use for ALL rows
                files = {}
                file_handle = None
                
                try:
                    # Open reference audio file
                    file_handle = open(ref_audio_path, 'rb')
                    content_type = ref_audio.content_type or 'audio/wav'
                    files = {'ref_audio': (os.path.basename(ref_audio_path), file_handle, content_type)}
                    
                    # Call the synthesize API with a timeout
                    logger.info(f"Calling synthesize API for row {index+1}, filename: {filename}")
                    
                    try:
                        response = requests.post(
                            f"{api_url}/synthesize",
                            data=payload,
                            files=files,
                            timeout=60  # Increased timeout for synthesis
                        )
                        
                        if response.status_code == 200:
                            # Save the audio file
                            audio_filename = f"{filename}"
                            if not audio_filename.lower().endswith(('.wav', '.mp3')):
                                audio_filename += '.wav'
                                
                            audio_path = os.path.join(current_app.config['UPLOAD_FOLDER'], audio_filename)
                            
                            with open(audio_path, 'wb') as f:
                                f.write(response.content)
                            
                            # Update result status
                            result["status"] = "success"
                            result["audio_path"] = f"/static/uploads/{audio_filename}"
                            
                            logger.info(f"Successfully synthesized audio for {filename}")
                        else:
                            # Handle error
                            try:
                                error_data = response.json()
                                error_message = error_data.get('message', error_data.get('error', 'Unknown error'))
                            except:
                                error_message = f"API returned status code {response.status_code}: {response.text[:200]}"
                            
                            result["status"] = "error"
                            result["error"] = error_message
                            logger.error(f"Synthesis error for {filename}: {error_message}")
                    
                    except requests.exceptions.RequestException as e:
                        result["status"] = "error"
                        result["error"] = f"Request failed: {str(e)}"
                        logger.error(f"Request exception for {filename}: {str(e)}")
                
                finally:
                    if file_handle:
                        try:
                            file_handle.close()
                        except:
                            pass
                
                # Add result to collection
                synthesis_results.append(result)
                
                # Add a delay between requests to avoid overwhelming the API
                if index < len(df) - 1:
                    logger.info(f"Waiting 3 seconds before processing next row...")
                    time.sleep(3)
            
            # Return all results
            return jsonify({
                "status": "success",
                "results": synthesis_results
            })
                
        except Exception as e:
            logger.exception("Error processing CSV")
            return jsonify({
                "status": "error",
                "message": f"Error processing CSV: {str(e)}"
            }), 400
            
    except Exception as e:
        logger.exception("Unexpected error processing CSV")
        return jsonify({
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500
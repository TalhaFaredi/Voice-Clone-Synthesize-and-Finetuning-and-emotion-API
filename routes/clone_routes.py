# routes/clone_routes.py

import logging
import os
from flask import Blueprint, request, jsonify, send_file
from flask import Blueprint, request, jsonify,current_app
from utils.file_utils import allowed_file
from config import get_api_url
import requests
import io
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
clone_routes = Blueprint('clone_routes', __name__)

# Configuration

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size




@clone_routes.route("/api/clone-voice", methods=["POST"])
def api_clone_voice():
    try:
        language = request.form.get("language")
        gen_text = request.form.get("gen_text")
        ref_audio_file_from_request = request.files.get("ref_audio_file")
        ref_audio_path_from_form = request.form.get("ref_audio_path")

        if not language:
            return jsonify({"error": "Language is required"}), 400
        if not gen_text:
            return jsonify({"error": "Text to speak (gen_text) is required"}), 400

        api_url = ""
        if language.lower() == "english":
            api_url = "http://34.238.66.156:5000/en/tts"
        elif language.lower() == "spanish":
            api_url = "http://34.238.66.156:5000/es/tts"
        else:
            return jsonify({"error": "Unsupported language specified"}), 400

        files_payload = None
        prepared_audio_file_tuple_for_requests = None
        
        # Ensure os is available if not already imported at the top ofclone_routes.py
        # For os.path.join, os.path.basename

        if ref_audio_file_from_request and ref_audio_file_from_request.filename != "":
            # Use the uploaded file directly
            # The key for the file part in the multipart/form-data request should be 'ref_audio'
            # as per the external API requirements mentioned by the user.
            prepared_audio_file_tuple_for_requests = (ref_audio_file_from_request.filename, ref_audio_file_from_request.stream, ref_audio_file_from_request.mimetype)
            files_payload = {"ref_audio": prepared_audio_file_tuple_for_requests}

        elif ref_audio_path_from_form:
            # Construct full path to the existing audio file
            # Assuming UPLOAD_FOLDER is defined and accessible asclone_routes.config['UPLOAD_FOLDER']
            # The path from the form is relative to the UPLOAD_FOLDER, e.g., "uploads/filename.wav"
            # We need the basename to join with UPLOAD_FOLDER
            base_filename = os.path.basename(ref_audio_path_from_form)
            full_audio_path = os.path.join(current_app.config["UPLOAD_FOLDER"], base_filename)
            
            if not os.path.exists(full_audio_path):
                return jsonify({"error": f"Reference audio file not found at path: {full_audio_path}"}), 404
            
            # The key for the file part should be 'ref_audio'
            files_payload = {"ref_audio": (base_filename, open(full_audio_path, "rb"), "application/octet-stream")}
            # Note: The file opened with open() will be closed by requests library after the request.

        else:
            return jsonify({"error": "No reference audio provided (file or path)"}), 400

        data_payload = {"gen_text": gen_text}
        
        logger.info(f"Calling external API: {api_url} with language: {language}")
        # For debugging, you might want to log payload details, but be careful with file contents.

        external_response = requests.post(api_url, files=files_payload, data=data_payload, timeout=120) # Increased timeout

        external_response.raise_for_status() # Will raise an HTTPError for bad responses (4XX or 5XX)

        if external_response.headers.get("Content-Type", "").startswith("audio/"):
            return send_file(
                io.BytesIO(external_response.content),
                mimetype=external_response.headers["Content-Type"],
                as_attachment=False # Send inline to be played by browser audio tag
            )
        else:
            logger.error(f"External API for {language} returned non-audio content: {external_response.headers.get('Content-Type')}. Response: {external_response.text[:200]}")
            return jsonify({"error": "External API returned non-audio content", "details": external_response.text[:500]}), 502

    except requests.exceptions.Timeout:
        logger.error(f"Timeout when calling external API for language {language}")
        return jsonify({"error": "Request to external voice cloning API timed out"}), 504 # Gateway Timeout
    except requests.exceptions.RequestException as e_req:
        logger.error(f"Error calling external API for language {language}: {str(e_req)}")
        return jsonify({"error": f"Failed to connect to external voice cloning API: {str(e_req)}"}), 502 # Bad Gateway
    except Exception as e:
        logger.error(f"Error in /api/clone-voice: {str(e)}")
        return jsonify({"error": f"Server error during voice cloning: {str(e)}"}), 500


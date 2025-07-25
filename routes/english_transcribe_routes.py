
import gc
import logging
import os
import sys
import shutil
import tempfile
import time
import torch
from werkzeug.utils import secure_filename
from importlib.resources import files
from flask import Blueprint, request, jsonify,current_app
from src.English_f5tts.English_train.English_utils.transcribe import create_metadata, transcribe_all

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"app_{time.strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)  # Optional: keep printing to console too
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.info(f"Logging configured. Logs will be written to {LOG_FILE}.")

english_transcribe_routes = Blueprint('english_transcribe_routes', __name__)

@english_transcribe_routes.route('/en/transcribe/<project_name>', methods=['POST'])
def english_transcribe_and_prepare(project_name):
    path_data = str(files("src").joinpath("./English_data"))
    project_name = project_name.lower().replace(" ", "_")  # Normalize project name to lowercase
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"API CALL: POST /transcribe/{project_name}")
    project_full_path = os.path.join(path_data, project_name)
    if not os.path.isdir(project_full_path):
        logger.warning(f"Project not found for transcription: {project_name} (expected at {project_full_path})")
        return jsonify({"error": f"Project '{project_name}' not found."}), 404

    language = request.args.get('language', 'English') # Default to English if not specified
    logger.info(f"Using language: {language} for project {project_name}")

    if 'voices' not in request.files:
        logger.warning("No 'voices' file part in the request for /transcribe")
        return jsonify({"error": "No 'voices' file part in the request. Please upload audio files under the 'voices' key."}), 400

    uploaded_files = request.files.getlist('voices')
    if not uploaded_files or all(f.filename == '' for f in uploaded_files):
        logger.warning("No selected files in 'voices' part for /transcribe")
        return jsonify({"error": "No files selected for upload."}), 400

    saved_file_paths = []
    # Use a subfolder within app.config['UPLOAD_FOLDER'] for this request to keep things tidy
    request_specific_temp_dir = tempfile.mkdtemp(prefix=f"{project_name}_", dir=current_app.config['UPLOAD_FOLDER1'])
    logger.info(f"Created temporary directory for uploads: {request_specific_temp_dir}")

    try:
        for file_storage_item in uploaded_files:
            if file_storage_item and file_storage_item.filename:
                filename = secure_filename(file_storage_item.filename)
                save_path = os.path.join(request_specific_temp_dir, filename)
                file_storage_item.save(save_path)
                saved_file_paths.append(save_path)
                logger.debug(f"Saved uploaded file: {save_path}")

        if not saved_file_paths:
             logger.warning("No valid files were actually saved from the upload.")
             shutil.rmtree(request_specific_temp_dir)
             return jsonify({"error": "No valid files uploaded after processing."}), 400

        logger.info(f"Calling transcribe_all for {len(saved_file_paths)} files in project '{project_name}'.")
        transcription_result_dict = transcribe_all(project_name, saved_file_paths, language, user_mode=False)

        if transcription_result_dict.get("status") == "error":
            logger.error(f"Transcription failed for project '{project_name}': {transcription_result_dict.get('message', 'Unknown error')}")
            shutil.rmtree(request_specific_temp_dir)
            logger.debug(f"Removed temporary directory: {request_specific_temp_dir}")
            gc.collect()
            return jsonify({"error": "Transcription failed", "details": transcription_result_dict.get('message')}), 500
        
        logger.info(f"Transcription for '{project_name}' finished with status: {transcription_result_dict.get('status')}. Summary: {transcription_result_dict.get('message')}")
        gc.collect()
        torch.cuda.empty_cache()
        # Determine tokenizer type - assuming pinyin based on original structure
        # This might need to come from project config or request if variable
        tokenizer_is_char_based = "pinyin" not in project_name.lower() # Heuristic, might need refinement
        logger.info(f"Calling create_metadata for project '{project_name}'. Deduced char_tokenizer={tokenizer_is_char_based}")
        prepare_result_dict, _ = create_metadata(project_name, ch_tokenizer=tokenizer_is_char_based)

        shutil.rmtree(request_specific_temp_dir)
        logger.debug(f"Removed temporary directory: {request_specific_temp_dir}")

        if prepare_result_dict.get("status") == "error":
            logger.error(f"Metadata preparation failed for project '{project_name}': {prepare_result_dict.get('message', 'Unknown error')}")
            return jsonify({"error": "Metadata preparation failed", "details": prepare_result_dict.get('message')}), 500

        logger.info(f"Metadata preparation for '{project_name}' finished with status: {prepare_result_dict.get('status')}. Summary: {prepare_result_dict.get('message')}")
        gc.collect()
        torch.cuda.empty_cache()
        return jsonify({
            "message": "Transcription and data preparation completed successfully.",
            "transcription_summary": transcription_result_dict,
            "preparation_summary": prepare_result_dict
        }), 200

    except Exception as e:
        logger.exception(f"An unexpected error occurred during transcription/preparation for project '{project_name}'")
        if 'request_specific_temp_dir' in locals() and os.path.exists(request_specific_temp_dir):
            shutil.rmtree(request_specific_temp_dir)
            logger.info(f"Removed temporary directory due to error: {request_specific_temp_dir}")
        return jsonify({"error": "An unexpected server error occurred", "details": str(e)}), 500
   
import logging
import os
import sys
import time
from flask import Blueprint, request, jsonify
from src.English_f5tts.English_train.English_utils.project_list import get_list_projects
from src.English_f5tts.English_train.English_utils.create_project import create_data_project


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
english_project_routes = Blueprint('english_project_routes', __name__)


@english_project_routes.route('/en/projects', methods=['GET'])
def get_english_projects():
    logger.info("API CALL: GET /projects")
    try:
        project_list, _ = get_list_projects()
        if project_list is None: # get_list_projects now returns [], None if path_data is bad
             logger.error("Failed to retrieve project list, path_data might be inaccessible.")
             return jsonify({"error": "Failed to retrieve project list or no projects found"}), 500
        return jsonify({"projects": project_list})
    except Exception as e:
        logger.exception("Unexpected error in GET /projects")
        return jsonify({"error": "An internal server error occurred"}), 500


@english_project_routes.route('/en/projects', methods=['POST'])
def create_english_project():
    logger.info("API CALL: POST /projects")
    if not request.is_json:
        logger.warning("Request is not JSON for POST /projects")
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    logger.debug(f"Request data: {data}")
    project_name = data.get('name').lower().replace(" ", "_") 
    tokenizer_type = data.get('tokenizer_type', 'pinyin') # Default to pinyin if not provided

    if not project_name:
        logger.warning("Missing 'name' in request body for POST /projects")
        return jsonify({"error": "Missing 'name' in request body"}), 400
    
    logger.info(f"Attempting to create project '{project_name}' with tokenizer '{tokenizer_type}'")
    created_project_actual_name = create_data_project(project_name, tokenizer_type)

    if created_project_actual_name:
        logger.info(f"Successfully created project: {created_project_actual_name}")
        return jsonify({"message": f"Project '{created_project_actual_name}' created/ensured successfully.", "project_name": created_project_actual_name}), 201
    else:
        logger.error(f"Failed to create project: {project_name}")
        return jsonify({"error": f"Failed to create project '{project_name}'"}), 500
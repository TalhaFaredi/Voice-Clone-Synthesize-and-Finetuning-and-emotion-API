
import gc
import logging
import os
import sys
import time
import torch
from importlib.resources import files
from flask import Blueprint, request, jsonify,current_app,Response


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

english_checkpoints_routes = Blueprint('english_checkpoints_routes', __name__)


@english_checkpoints_routes.route('/en/list-checkpoints', methods=['POST'])
def list_checkpoints_for_english_project():
    try:
        path_project_ckpts = str(files("src").joinpath("./English_ckpts"))
        gc.collect()
        torch.cuda.empty_cache()
        data = request.get_json()
        project_name = data.get("project_name").lower().replace(" ", "_")

        if not project_name:
            return jsonify({"error": "Missing 'project_name' in payload"}), 400

        project_ckpt_path = os.path.join(path_project_ckpts, project_name)
        if not os.path.isdir(project_ckpt_path):
            return jsonify({"error": f"Project folder not found: {project_ckpt_path}"}), 404

        ckpt_files = [
            f for f in os.listdir(project_ckpt_path)
            if os.path.isfile(os.path.join(project_ckpt_path, f)) and f.endswith(".pt")
        ]

        return jsonify({
            "project": project_name,
            "checkpoints": ckpt_files
        }), 200

    except Exception as e:
        logger.error(f"Error in /list-checkpoints: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


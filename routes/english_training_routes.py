
import gc
import logging
import os
import sys
import time
import torch
from flask import Blueprint, request, jsonify,current_app,Response
from services.english_training import start_english_training

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

english_training_routes = Blueprint('english_training_routes', __name__)
@english_training_routes.route('/en/start-training', methods=['POST'])
def english_training():
    data = request.json
    def wrapped():
        yield from start_english_training(data)
        # Garbage collect after training
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("Garbage collected after training.")
    return Response(wrapped(), mimetype='text/event-stream')


import gc
import logging
import os
import sys
import time
import torch
from flask import Blueprint, request, jsonify,current_app,Response
from services.spanish_training import start_spanish_training

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

spanish_training_routes = Blueprint('spanish_training_routes', __name__)

@spanish_training_routes.route('/es/start-training', methods=['POST'])
def spanish_training():
    data = request.json
    def wrapped():
        yield from start_spanish_training(data)
        # Garbage collect after training
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("Garbage collected after training.")
    return Response(wrapped(), mimetype='text/event-stream')



import gc
import json
import logging
from multiprocessing import Process
import os
import sys
import time
import torch

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

def start_spanish_training(data):
    try:
        project_name = data.get("project_name")
        if not project_name:
            error_msg = {"error": "Missing 'project_name' in payload"}
            # logger.error("Missing 'project_name' in payload")
            logger.error(error_msg["error"])
            yield f"data: {json.dumps(error_msg)}\n\n"
            # yield jsonify({"error": "Missing 'project_name' in payload"})
            return

        # Run training in subprocess to ensure full GPU memory release
        def train_target():
            try:
                from src.Spanish_f5tts.Spanish_train.Spanish_utils.spanish import spanish_run_training
                for message in spanish_run_training(project_name):
                    print(f"[Train] {message}")
            except Exception as e:
                print(f"[Train Error] {e}")

        p = Process(target=train_target)
        p.start()
        p.join()  # Wait until training completes

        yield "data: Training completed.\n\n"
        logger.info("Training subprocess finished.")
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in start_training: {str(e)}")
        yield f"data: Error: {str(e)}\n\n"
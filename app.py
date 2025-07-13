import os
import sys
import time
import platform
import tempfile
from flask_cors import CORS
import logging
from routes.english_infer_routes import english_infer_routes
from routes.spanish_infer_routes import spanish_infer_routes
from routes.english_project_routes import english_project_routes
from routes.spanish_project_routes import spanish_project_routes
from routes.english_transcribe_routes import english_transcribe_routes
from routes.spanish_transcribe_routes import spanish_transcribe_routes
from routes.spanish_training_routes import spanish_training_routes
from routes.english_checkpoints_routes import english_checkpoints_routes
from routes.spanish_checkpoints_routes import spanish_checkpoints_routes
from routes.english_synthesize_routes import english_synthesize_routes
from routes.spanish_synthesize_routes import spanish_synthesize_routes
from routes.english_training_routes import english_training_routes
from routes.emotions_routes import emotions_routes
from flask import Flask,render_template
from services.db_service import init_db
from routes.profile_routes import profile_routes
from routes.clone_routes import clone_routes
from routes.finetune_routes import finetune_routes
from routes.project_routes import project_routes
from routes.training_routes import training_routes
from routes.synthesize_routes import synthesize_routes
from routes.process_csv_routes import process_csv_routes

app = Flask(__name__, static_folder='static',template_folder='templates')
CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



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


# app = Flask(__name__)
# CORS(app)
system = platform.system()


python_executable = sys.executable or "python"

UPLOAD_FOLDER1 = os.path.join(tempfile.gettempdir(), 'f5_uploads')


if not os.path.exists(UPLOAD_FOLDER1):
    os.makedirs(UPLOAD_FOLDER1, exist_ok=True)
app.config['UPLOAD_FOLDER1'] = UPLOAD_FOLDER1
# UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'f5_uploads')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     logger.info(f"Created upload folder: {UPLOAD_FOLDER}")


# Register Blueprints
app.register_blueprint(english_infer_routes)
app.register_blueprint(spanish_infer_routes)
app.register_blueprint(english_project_routes)
app.register_blueprint(spanish_project_routes)
app.register_blueprint(english_transcribe_routes)
app.register_blueprint(spanish_transcribe_routes)
app.register_blueprint(english_training_routes)
app.register_blueprint(spanish_training_routes)
app.register_blueprint(english_checkpoints_routes)
app.register_blueprint(spanish_checkpoints_routes)
app.register_blueprint(english_synthesize_routes)
app.register_blueprint(spanish_synthesize_routes)
app.register_blueprint(emotions_routes)
app.register_blueprint(profile_routes)
app.register_blueprint(clone_routes)
app.register_blueprint(finetune_routes)
app.register_blueprint(project_routes)
app.register_blueprint(synthesize_routes)
app.register_blueprint(training_routes)
app.register_blueprint(process_csv_routes)


@app.route('/')
def index():
    return render_template('index.html') # Assumes fixed_index.html is renamed to index.html
if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
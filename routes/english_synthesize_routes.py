
import gc
import logging
import os
import random
import sys
import tempfile
import time
import torch
import shutil
from werkzeug.utils import secure_filename
from importlib.resources import files
from speechbrain.pretrained import SpeakerRecognition
from flask import Blueprint, request, jsonify,after_this_request,send_from_directory
from src.English_f5tts.English_train.English_utils.inference import infer
from scipy.io import wavfile
from pydub import AudioSegment

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

english_synthesize_routes = Blueprint('english_synthesize_routes', __name__)


@english_synthesize_routes.route("/en/synthesize", methods=["POST"])
def english_synthesize():
    gc.collect()
    torch.cuda.empty_cache()
    temp_dir_ref_audio = None
    generated_audio_output_path = None # Path of the file returned by infer()
    path_project_ckpts = str(files("src").joinpath("./English_ckpts"))
   
    @after_this_request
    def cleanup(response):
        nonlocal temp_dir_ref_audio, generated_audio_output_path # Ensure access to outer scope variables
        if generated_audio_output_path and os.path.exists(generated_audio_output_path):
            try:
                os.remove(generated_audio_output_path)
                logger.info(f"Cleaned up generated audio file: {generated_audio_output_path}")
            except Exception as e:
                logger.error(f"Error deleting generated audio file {generated_audio_output_path}: {e}")
        if temp_dir_ref_audio and os.path.isdir(temp_dir_ref_audio):
            try:
                shutil.rmtree(temp_dir_ref_audio)
                logger.info(f"Cleaned up temp ref audio directory: {temp_dir_ref_audio}")
            except Exception as e:
                logger.error(f"Error deleting temp ref audio directory {temp_dir_ref_audio}: {e}")
        return response

    try:
        project_select = request.form.get("project_select")
        checkpoint = request.form.get("checkpoint")
        gen_text = request.form.get("gen_text")
        ref_audio_file = request.files.get("ref_audio")
        
        gen_text += ". "
        # Validate mandatory fields
        if not project_select:
            return jsonify({"error": "Missing required field: project_select"}), 400
        if not checkpoint:
            return jsonify({"error": "Missing required field: checkpoint"}), 400
        if not gen_text:
            return jsonify({"error": "Missing required field: gen_text"}), 400
        if not ref_audio_file:
            return jsonify({"error": "Missing required file: ref_audio"}), 400

        exp_name = "F5TTS_v1_Base" # As per original script logic

        # Get optional parameters with defaults
        speed = float(request.form.get("speed", 0.9))
        seed = int(request.form.get("seed", -1))  # -1 for random in infer function
        nfe_step = int(request.form.get("nfe_step", 32))
        use_ema = request.form.get("use_ema", "false").lower() == "true"
        remove_silence = request.form.get("remove_silence", "false").lower() == "true"
        ref_text = request.form.get("ref_text", "") # Default to empty string

        checkpoint_path = os.path.join(path_project_ckpts, project_select, checkpoint)
        if not os.path.isfile(checkpoint_path):
            logger.error(f"Checkpoint not found at {checkpoint_path} for project {project_select}")
            return jsonify({"error": f"Checkpoint not found: {checkpoint}"}), 404

        # Securely save uploaded reference audio to a temporary location
        temp_dir_ref_audio = tempfile.mkdtemp()
        ref_audio_filename = secure_filename(ref_audio_file.filename)
        sample_file_directory = str(files("src").joinpath(f"./English_data/{project_select}/wavs"))
        print(sample_file_directory)
        wav_files = [f for f in os.listdir(sample_file_directory) if f.endswith(".wav")]
        # Function to get the duration of an audio file in seconds
        def get_audio_duration(audio_file):
            audio = AudioSegment.from_wav(audio_file)
            return len(audio) / 1000  # Convert milliseconds to seconds

        

        # Filter files that are >= 5 seconds
        valid_audio_files = []
        for file in wav_files:
            file_path = os.path.join(sample_file_directory, file)
            duration = get_audio_duration(file_path)
            if duration >= 5:
                valid_audio_files.append(file_path)

        # Print the valid audio files
        print("Audio files with duration >= 10 seconds:")
        for file in valid_audio_files:
            print(file)
        # Check if any .wav files exist
        # if not wav_files:
        #     print("No .wav files found in the directory.")
        # else:
            # Pick a random .wav file
        random_file = random.choice(valid_audio_files)
        sample_file= os.path.join(sample_file_directory, random_file)

        # Load the WAV file
        sample_rate, audio_data = wavfile.read(sample_file)

        # Output information
        print(f"\nLoaded file: {random_file}")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Audio Data Shape: {audio_data.shape}")
        ref_audio_path = os.path.join(temp_dir_ref_audio, ref_audio_filename)
        ref_audio_file.save(ref_audio_path)
        verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )


        file1 = sample_file
        file2 = ref_audio_path


        score, prediction = verification.verify_files(file1, file2)

        print("Similarity score:", score.item())
        print("Same speaker?", "Yes" if prediction.item() == 1 else "No")
        if prediction.item() == 1:
            
            logger.info(f"Saved reference audio to temporary path: {ref_audio_path}")

            # Corrected call to the infer function
            infer_result_tuple = infer(
                project=project_select,
                file_checkpoint=checkpoint_path,
                exp_name=exp_name,
                ref_text=ref_text,
                ref_audio=ref_audio_path,
                gen_text=f"{gen_text.lower()} .",
                nfe_step=nfe_step,
                use_ema=use_ema,
                speed=speed,
                seed=seed,
                remove_silence=remove_silence
            )
            
            # infer returns (audio_path, device_str, seed_str) or (None, error_msg, None) on checkpoint error
            if infer_result_tuple[0] is None and infer_result_tuple[1] == "checkpoint not found!":
                return jsonify({"error": "Inference failed: checkpoint not found internally."}), 404

            generated_audio_output_path = infer_result_tuple[0]
            # device_info = infer_result_tuple[1]
            # seed_info = infer_result_tuple[2]

            logger.info(f"Sending generated audio file: {generated_audio_output_path}")
            gc.collect()
            torch.cuda.empty_cache()
            return send_from_directory(
                directory=os.path.dirname(generated_audio_output_path),
                path=os.path.basename(generated_audio_output_path),
                mimetype="audio/wav",
                as_attachment=True,
                download_name=f"{secure_filename(project_select)}_{secure_filename(gen_text[:20])}.wav"
            )
        else:
            return jsonify({"error": "Please provide your original audio"}), 500

    except Exception as e:
        logger.error(f"Error in /synthesize endpoint: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

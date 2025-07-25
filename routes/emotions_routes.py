import gc
import logging
import os
import re
import sys
import tempfile
import time
import soundfile as sf
from services.emotions import emotions
from num2words import num2words
from flask import Blueprint, request, jsonify, send_file
from src.English_f5tts.English_train.English_utils.inference import infer


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

emotions_routes = Blueprint('emotions_routes', __name__)

@emotions_routes.route('/api/generate-voice', methods=['POST'])
def generate_voice_endpoint():
    try:
        DEFAULT_TTS_MODEL = "F5-TTS_v1"
        # Get file and form fields
        if 'ref_audio' not in request.files:
            return jsonify({"error": "Reference audio file is required"}), 400

        ref_audio_file = request.files['ref_audio']
        ref_text = request.form.get('ref_text', '')
        gen_text1 = request.form.get('gen_text', '')+'.'
        language = request.form.get('language', '')
        
        
        emotion_name = request.form.get('emotion_name', 'Regular')
        emotion_seed = int(request.form.get('emotion_seed', -1))
        emotion_speed = float(request.form.get('emotion_speed', 0.9))
        remove_silence = request.form.get('remove_silence', 'true').lower() == 'true'
        # Create the JSON metadata line
        metadata_line = f'{{"name": "{emotion_name}", "seed": {emotion_seed}, "speed": {emotion_speed}}}'
        
        # Create the formatted emotion line
        emotion_line = f'{{{emotion_name}}} {gen_text1}' if gen_text1 else ''
        
        # Combine them into final output
        if emotion_line:
            gen_text = f"{metadata_line}\n{emotion_line}"
        else:
            gen_text = metadata_line  # or handle empty gen_text case as needed
        
        print(gen_text)
        if not gen_text:
            return jsonify({"error": "Text to generate is required"}), 400

        # Parse gen_text to remove metadata sections
        def traducir_numero_a_texto(texto):
            texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
            texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)
            
            def reemplazar_numero(match):
                numero = match.group()
                return num2words(int(numero), lang='es')
            
            texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)
            
            return texto_traducido
        def parse_gen_text(text):
            # Regular expression to match metadata sections like {"name": "Angry", "seed": -1, "speed": 1}
            pattern = r"\{.*?\}"
            # Split the text by the pattern and filter out empty strings
            segments = [segment.strip() for segment in re.split(pattern, text) if segment.strip()]
            return " ".join(segments)
        if not gen_text.startswith(" "):
            gen_text = " " + gen_text
        if not gen_text.endswith(". "):
            gen_text += ". "
        gen_text = gen_text.lower()
        gen_text = traducir_numero_a_texto(gen_text)
        # Clean the gen_text input
        cleaned_gen_text = parse_gen_text(gen_text)

        # Save the uploaded audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            ref_audio_file.save(f)
            ref_audio_path = f.name
        
        # Call the voice generation logic
        result, spectrogram_path, _, used_seed = emotions(
            ref_audio_path,
            ref_text,
            cleaned_gen_text,  # Use the cleaned text
            DEFAULT_TTS_MODEL,
            language,
            remove_silence,
            emotion_seed,
            speed=emotion_speed
            
        )

        if result is None:
            return jsonify({"error": "Voice generation failed"}), 500

        sample_rate, audio_data = result

        # Save the output audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, audio_data, sample_rate)
            output_path = f.name
       
        # Send the file directly
        return send_file(
            output_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='generated.wav'
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Request failed: {str(e)}"}), 500


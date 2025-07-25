
import gc
import tempfile
import torch
import soundfile as sf
from flask import Blueprint, request, jsonify,send_file
from services.spanish_infer import spanish_infer

spanish_infer_routes = Blueprint('spanish_infer_routes', __name__)


@spanish_infer_routes.route('/es/tts', methods=['POST'])
def spanish_tts():
    # Check that both required inputs are provided:
    if 'ref_audio' not in request.files:
        return jsonify({"error": "Missing 'ref_audio' file"}), 400
    if 'gen_text' not in request.form:
        return jsonify({"error": "Missing 'gen_text' parameter"}), 400

    ref_audio_file = request.files['ref_audio']
    gen_text = request.form['gen_text']
    gen_text += ". "

    # Save the uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        ref_audio_path = tmp.name
        ref_audio_file.save(ref_audio_path)

    # Use an empty string for ref_text to auto-transcribe it
    ref_text = ""

    # Set default parameters (other values are not exposed to the API)
    model_choice = "F5-TTS"  # Not actively used in infer() except for consistency
    remove_silence = False
    cross_fade_duration = 0.15
    model_choice = "F5-TTS"

    cross_fade_duration = 0.15
    nfe_step=32

    # Get speed from form-data if provided, otherwise default to 0.9
    try:
        speed = float(request.form.get("speed", 0.9))
    except ValueError:
        return jsonify({"error": "Invalid 'speed' parameter. Must be a number."}), 400


    try:
        # Call the infer function with only the required parameters
        audio_result, spectrogram_path = spanish_infer(
            ref_audio_path,
            ref_text,
            f"{gen_text.lower()} .",
            model_choice,
            remove_silence,
            nfe_step,
            cross_fade_duration,
            speed,
            # show_info=dummy  # using dummy info object
        )
    except Exception as e:
        return jsonify({"error": f"Synthesis failed: {str(e)}"}), 500

    sample_rate, wave = audio_result

    # Write the resulting synthesized audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_audio_file:
        out_path = out_audio_file.name
        sf.write(out_path, wave, sample_rate)

    gc.collect()
    torch.cuda.empty_cache()
    # Return the synthesized audio file as a download
    return send_file(out_path, mimetype="audio/wav", as_attachment=True, download_name="cloned_audio.wav")

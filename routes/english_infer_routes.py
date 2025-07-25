import gc
import tempfile
import torch
import soundfile as sf
from flask import Blueprint, request, jsonify,send_file
from services.english_infer import english_infer


english_infer_routes = Blueprint('english_infer_routes', __name__)

@english_infer_routes.route('/en/tts', methods=['POST'])
def english_tts():
    if 'ref_audio' not in request.files or 'gen_text' not in request.form:
        return jsonify({"error": "Missing 'ref_audio' or 'gen_text'"}), 400

    ref_audio_file = request.files['ref_audio']
    gen_text = request.form['gen_text']
    gen_text = gen_text.lower() + ". "

    ref_text = request.form.get('ref_text', '')
    remove_silence = request.form.get('remove_silence', 'false').lower() == 'true'
    cross_fade_duration = float(request.form.get('cross_fade_duration', 0.15))
    nfe_step = int(request.form.get('nfe_step', 32))
    speed = float(request.form.get('speed', 0.9))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        ref_audio_path = tmp.name
        ref_audio_file.save(ref_audio_path)

    try:
        audio_result, spectrogram_path =english_infer(
            ref_audio_path,
            ref_text,
            f"{gen_text.lower()} .",
            model="F5-TTS",
            remove_silence=remove_silence,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
        )
    except Exception as e:
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500

    sample_rate, wave = audio_result
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_audio_file:
        out_path = out_audio_file.name
        sf.write(out_path, wave, sample_rate)

    gc.collect()
    torch.cuda.empty_cache()
    
    return send_file(out_path, mimetype="audio/wav", as_attachment=True, download_name="tts_output.wav")

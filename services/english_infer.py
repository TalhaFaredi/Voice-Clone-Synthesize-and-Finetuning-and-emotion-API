import json
import tempfile
import torchaudio
import soundfile as sf
from cached_path import cached_path
from src.English_f5tts.model import DiT
from src.English_f5tts.infer.utils_infer import (load_vocoder,load_model,preprocess_ref_audio_text,infer_process,remove_silence_for_generated_wav,save_spectrogram)


DEFAULT_ENGLISH_TTS_MODEL = "F5-TTS_v1"
# Load default model config
DEFAULT_ENGLISH_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

vocoder = load_vocoder()
F5TTS_ENGLISH_ema_model = load_model(
    DiT,
    json.loads(DEFAULT_ENGLISH_TTS_MODEL_CFG[2]),
    str(cached_path(DEFAULT_ENGLISH_TTS_MODEL_CFG[0])),
)



def english_infer(ref_audio_orig,ref_text,gen_text,model,remove_silence,cross_fade_duration=0.15,nfe_step=32,speed=0.9):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text)
    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        F5TTS_ENGLISH_ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
    )

    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


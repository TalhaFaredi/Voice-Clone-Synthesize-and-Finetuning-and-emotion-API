

import json
import tempfile
import torch
import numpy as np
from functools import lru_cache
from cached_path import cached_path
import soundfile as sf
import torchaudio
from importlib.resources import files
from src.English_f5tts.model import DiT
from src.English_f5tts.infer.utils_infer import (load_vocoder,load_model,preprocess_ref_audio_text,infer_process,remove_silence_for_generated_wav,save_spectrogram)

DEFAULT_EMOTIONS_TTS_MODEL = "F5-TTS_v1"
DEFAULT_English_EMOTIONS_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]
DEFAULT_Spanish_EMOTIONS_TTS_MODEL_CFG = [
    str(files("src").joinpath("./weights/pruned_model.safetensors")),  # HF model path
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",           
    # Vocabulary file
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),  # Model config
]
emotions_vocoder = load_vocoder()

def load_f5tts(language):
    if language.lower() == "spanish":
        ckpt_path = str(cached_path(DEFAULT_Spanish_EMOTIONS_TTS_MODEL_CFG[0]))
        F5TTS_model_cfg = json.loads(DEFAULT_Spanish_EMOTIONS_TTS_MODEL_CFG[2])
    elif language.lower()=='english':  # Default to English
        ckpt_path = str(cached_path(DEFAULT_English_EMOTIONS_TTS_MODEL_CFG[0]))
        F5TTS_model_cfg = json.loads(DEFAULT_English_EMOTIONS_TTS_MODEL_CFG[2])
    else:
        raise ValueError(f"Unsupported language: {language}. Supported languages are 'english' and 'spanish'.")
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


@lru_cache(maxsize=100)
def emotions(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    language,
    remove_silence,
    seed,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1
    # show_info=None,
    
    ):
    F5TTS_emotions_ema_model = load_f5tts(language)
    if not ref_audio_orig:
        return None, None, ref_text, seed

    # Set inference seed
    if seed < 0 or seed > 2**31 - 1:
        seed = np.random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    used_seed = seed

    if not gen_text.strip():
        return None, None, ref_text, used_seed

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text)
    
    if model == DEFAULT_EMOTIONS_TTS_MODEL:
        ema_model = F5TTS_emotions_ema_model
    # elif model == "E2-TTS":
    #     global E2TTS_ema_model
    #     if E2TTS_ema_model is None:
    #         if show_info:
    #             show_info("Loading E2-TTS model...")
    #         E2TTS_ema_model = load_e2tts()
    #     ema_model = E2TTS_ema_model
    # elif isinstance(model, tuple) and model[0] == "Custom":
    #     global custom_ema_model, pre_custom_path
    #     if pre_custom_path != model[1]:
    #         if show_info:
    #             show_info("Loading Custom TTS model...")
    #         custom_ema_model = load_custom(model[1], vocab_path=model[2], model_cfg=model[3])
    #         pre_custom_path = model[1]
    #     ema_model = custom_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        emotions_vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text, used_seed



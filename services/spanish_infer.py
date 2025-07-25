

import re
import tempfile
import torchaudio
import soundfile as sf
from num2words import num2words
from importlib.resources import files
from src.Spanish_f5tts.model import spanish_dit, spanish_unett
from src.Spanish_f5tts.infer.utils_infer import (spanish_load_vocoder,load_spanish_model,preprocess_spanish_ref_audio_text,spanish_infer_process,spanish_remove_silence_for_generated_wav,save_spanish_spectrogram)

spanish_vocoder = spanish_load_vocoder()

# Load the F5-TTS model
F5TTS_SPANISH_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

ckpt_path = str(files("src").joinpath("./weights/pruned_model.safetensors"))


F5TTS_SPANISH_ema_model = load_spanish_model(
    spanish_dit, F5TTS_SPANISH_model_cfg, ckpt_path)


def traducir_numero_a_texto(texto):
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)
    
    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')
    
    texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)
    
    return texto_traducido


def spanish_infer(ref_audio_orig, ref_text, gen_text, model, remove_silence,nfe_step,cross_fade_duration=0.15, speed=0.9):
    # Automatically transcribe the reference audio if ref_text is empty
    ref_audio, ref_text = preprocess_spanish_ref_audio_text(ref_audio_orig, ref_text)

    ema_model = F5TTS_SPANISH_ema_model

    # Preprocess gen_text (adding needed padding and punctuation)
    if not gen_text.startswith(" "):
        gen_text = " " + gen_text
    if not gen_text.endswith(". "):
        gen_text += ". "
    gen_text = gen_text.lower()
    gen_text = traducir_numero_a_texto(gen_text)

    # Synthesize the voice, using dummy progress in place of Gradio's Progress
    final_wave, final_sample_rate, combined_spectrogram = spanish_infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        spanish_vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        #show_info=show_info,
        # progress=dummy,  # dummy progress object
    )

    # Remove silence if needed
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            spanish_remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram image (optional, if you want to also return it)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spanish_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


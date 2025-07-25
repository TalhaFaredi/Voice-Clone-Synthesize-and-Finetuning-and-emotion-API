from importlib.resources import files
import platform
import sys
import tempfile
from src.Spanish_f5tts.api  import F5TTS
from asyncio.log import logger
import os



training_process = None
system = platform.system()
python_executable = sys.executable or "python"
tts_api = None
last_checkpoint = ""
last_device = ""
last_ema = None
stop_signal = False
path_data = str(files("src").joinpath("./Spanish_data"))
path_project_ckpts = str(files("src").joinpath("./Spanish_ckpts"))
file_train = str(files("src.Spanish_f5tts.Spanish_train").joinpath("finetune_cli.py"))


def infer_spanish(
    project, file_checkpoint, exp_name, ref_text, ref_audio, gen_text, nfe_step, use_ema, speed, seed, remove_silence
):
    global last_checkpoint, last_device, tts_api, last_ema, training_process, path_data

    if not os.path.isfile(file_checkpoint):
        # It's better to raise an error or return a specific error indicator
        logger.error(f"Checkpoint not found: {file_checkpoint}")
        return None, "checkpoint not found!", None # Match tuple structure

    # device_test logic might need adjustment based on 'training_process'
    device_test = "cpu" if training_process is not None else None 

    if last_checkpoint != file_checkpoint or last_device != device_test or last_ema != use_ema or tts_api is None:
        if last_checkpoint != file_checkpoint:
            last_checkpoint = file_checkpoint
        if last_device != device_test:
            last_device = device_test
        if last_ema != use_ema:
            last_ema = use_ema
        
        # 'path_data' is crucial here
        vocab_file = os.path.join(path_data, project, "vocab.txt")
        if not os.path.isfile(vocab_file):
            logger.error(f"Vocabulary file not found: {vocab_file}")
            # Consider how to handle this error; F5TTS might fail.
            # For now, F5TTS will likely raise an error if vocab_file is missing.

        logger.info(f"Initializing F5TTS with: model={exp_name}, ckpt={file_checkpoint}, vocab={vocab_file}, device={device_test}, ema={use_ema}")
        tts_api = F5TTS(
            model=exp_name, ckpt_file=file_checkpoint, vocab_file=vocab_file, device=device_test, use_ema=use_ema
        )
        logger.info(f"F5TTS API updated. Device: {device_test}, Checkpoint: {file_checkpoint}, EMA: {use_ema}")

    if seed == -1:  # -1 used for random
        actual_seed = None
    else:
        actual_seed = seed

    # Create a temporary file for the output, ensuring it's deleted after.
    # The infer method of tts_api writes to f.name.
    # We will return f.name and let the caller (synthesize route) handle sending and cleanup.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f_out:
        output_wav_path = f_out.name
    
    try:
        tts_api.infer(
            ref_file=ref_audio,
            ref_text=ref_text.lower().strip(),
            gen_text=gen_text.lower().strip(),
            nfe_step=nfe_step,
            speed=speed,
            remove_silence=remove_silence,
            file_wave=output_wav_path,
            seed=actual_seed,
        )
        # tts_api.seed will be updated by the infer call if seed was None
        returned_seed = tts_api.seed if hasattr(tts_api, 'seed') else actual_seed 
        logger.info(f"Inference successful. Output: {output_wav_path}, Device: {tts_api.device}, Seed used: {returned_seed}")
        return output_wav_path, str(tts_api.device), str(returned_seed)
    except Exception as e:
        logger.error(f"Error during tts_api.infer: {e}", exc_info=True)
        if os.path.exists(output_wav_path):
            os.remove(output_wav_path) # Clean up temp file on error
        raise # Re-raise the exception to be caught by the route

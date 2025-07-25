from asyncio.log import logger
from glob import glob
import json
import os
from importlib.resources import files
import logging
import shutil # Added for logging
import librosa
import torchaudio
from Spanish_f5tts.model.utils import convert_char_to_pinyin
from datasets.arrow_writer import ArrowWriter
import numpy as np
from scipy.io import wavfile
from Spanish_f5tts.infer.utils_infer import spanish_transcribe


path_data = str(files("src").joinpath("./Spanish_data"))
path_project_ckpts = str(files("src").joinpath("./Spanish_ckpts"))
file_train = str(files("src.Spanish_f5tts.Spanish_train").joinpath("finetune_cli.py"))



logging.basicConfig(level=logging.INFO)



def get_audio_duration(audio_path):
    try:
        audio, sample_rate = torchaudio.load(audio_path)
        duration = audio.shape[1] / sample_rate
        return duration
    except Exception as e:
        print(f"Error getting duration for {audio_path}: {e}", exc_info=True)
        raise # Re-raise the exception to be caught by the caller



def clear_text(text):
    # logger.debug(f"Clearing text: '{text}'")
    cleared = text.lower().strip()
    # logger.debug(f"Cleared text: '{cleared}'")
    return cleared



def get_rms(y, frame_length=2048, hop_length=512, pad_mode="constant"):
    # logger.debug(f"Calculating RMS for audio data... Shape: {y.shape if isinstance(y, np.ndarray) else 'Unknown'}") # Too verbose for every call
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)
    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    rms = np.sqrt(power)
    # logger.debug("RMS calculation complete.")
    return rms




class Slicer:
    def __init__(self, sr: int, threshold: float = -40.0, min_length: int = 2000, min_interval: int = 300,
                 hop_size: int = 20, max_sil_kept: int = 2000):
        logger.debug(f"Initializing Slicer: sr={sr}, threshold={threshold}, min_length={min_length}ms, ...")
        if not min_length >= min_interval >= hop_size:
            logger.error("SlicerInitializationError: min_length >= min_interval >= hop_size must be satisfied.")
            raise ValueError("The following condition must be satisfied: min_length >= min_interval >= hop_size")
        if not max_sil_kept >= hop_size:
            logger.error("SlicerInitializationError: max_sil_kept >= hop_size must be satisfied.")
            raise ValueError("The following condition must be satisfied: max_sil_kept >= hop_size")
        min_interval_samples = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval_samples), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size) # in hops
        self.min_interval = round(min_interval_samples / self.hop_size) # in hops
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size) # in hops
        logger.info(f"Slicer initialized: hop_size_samples={self.hop_size}, win_size_samples={self.win_size}, min_length_hops={self.min_length}, min_interval_hops={self.min_interval}, max_sil_kept_hops={self.max_sil_kept}")

    def _apply_slice(self, waveform, begin_hop, end_hop):
        # logger.debug(f"Applying slice from hop {begin_hop} to {end_hop}")
        start_sample = begin_hop * self.hop_size
        end_sample = min(waveform.shape[-1], end_hop * self.hop_size)
        if len(waveform.shape) > 1:
            return waveform[:, start_sample : end_sample]
        else:
            return waveform[start_sample : end_sample]

    def slice(self, waveform):
        logger.debug(f"Slicing waveform of shape {waveform.shape}")
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
            logger.debug("Multi-channel audio detected, using mean for RMS calculation.")
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length * self.hop_size: # Compare samples with min_length in samples
            logger.info("Waveform is shorter than min_length, returning as a single chunk.")
            return [[waveform, 0, samples.shape[0]]]
        
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        logger.debug(f"RMS list calculated, shape: {rms_list.shape}")
        sil_tags = []
        silence_start_hop = None
        clip_start_hop = 0
        
        for i, rms_val in enumerate(rms_list):
            if rms_val < self.threshold:
                if silence_start_hop is None:
                    silence_start_hop = i
                continue
            if silence_start_hop is None:
                continue
            
            is_leading_silence = silence_start_hop == 0 and i > self.max_sil_kept
            need_slice_middle = (i - silence_start_hop >= self.min_interval and 
                                 i - clip_start_hop >= self.min_length)
            
            if not is_leading_silence and not need_slice_middle:
                silence_start_hop = None
                continue
            
            # Found a silence segment to process
            # logger.debug(f"Processing silence segment: start_hop={silence_start_hop}, end_hop={i}, clip_start_hop={clip_start_hop}")
            if i - silence_start_hop <= self.max_sil_kept: # Short silence, find minimum RMS within
                pos = rms_list[silence_start_hop : i + 1].argmin() + silence_start_hop
                # logger.debug(f"Short silence. Min RMS at hop {pos}")
                if silence_start_hop == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start_hop = pos
            elif i - silence_start_hop <= self.max_sil_kept * 2: # Medium silence, find min at both ends
                pos_l = rms_list[silence_start_hop : silence_start_hop + self.max_sil_kept + 1].argmin() + silence_start_hop
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + (i - self.max_sil_kept)
                # logger.debug(f"Medium silence. Min RMS at hops {pos_l} (left) and {pos_r} (right)")
                if silence_start_hop == 0:
                    sil_tags.append((0, pos_r))
                    clip_start_hop = pos_r
                else:
                    sil_tags.append((pos_l, pos_r))
                    clip_start_hop = pos_r
            else: # Long silence, cut from both ends
                pos_l = rms_list[silence_start_hop : silence_start_hop + self.max_sil_kept + 1].argmin() + silence_start_hop
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + (i - self.max_sil_kept)
                # logger.debug(f"Long silence. Min RMS at hops {pos_l} (left) and {pos_r} (right). Will create a cut.")
                if silence_start_hop == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start_hop = pos_r # The new audio segment starts after this cut
            silence_start_hop = None

        total_frames_hops = rms_list.shape[0]
        if silence_start_hop is not None and total_frames_hops - silence_start_hop >= self.min_interval: # Trailing silence
            silence_end_hop = min(total_frames_hops, silence_start_hop + self.max_sil_kept)
            pos = rms_list[silence_start_hop : silence_end_hop + 1].argmin() + silence_start_hop
            # logger.debug(f"Trailing silence. Min RMS at hop {pos}. End of audio at hop {total_frames_hops}")
            sil_tags.append((pos, total_frames_hops)) # Keep audio until the very end
        
        chunks = []
        if not sil_tags:
            logger.info("No suitable silence tags found for slicing, returning original waveform as one chunk.")
            return [[waveform, 0, total_frames_hops * self.hop_size]]
        else:
            logger.debug(f"Found silence tags: {sil_tags}")
            current_hop = 0
            if sil_tags[0][0] > 0: # Content before the first silence tag
                # logger.debug(f"Adding chunk from hop 0 to {sil_tags[0][0]}")
                chunks.append([self._apply_slice(waveform, 0, sil_tags[0][0]), 0, sil_tags[0][0] * self.hop_size])
            
            for i in range(len(sil_tags)):
                start_slice_hop = sil_tags[i][1] # Start audio after the current silence
                end_slice_hop = sil_tags[i+1][0] if i + 1 < len(sil_tags) else total_frames_hops # End audio before next silence or end of audio
                
                if start_slice_hop < end_slice_hop : # Ensure there's content between silences
                    # logger.debug(f"Adding chunk from hop {start_slice_hop} to {end_slice_hop}")
                    chunks.append([
                        self._apply_slice(waveform, start_slice_hop, end_slice_hop),
                        start_slice_hop * self.hop_size,
                        end_slice_hop * self.hop_size,
                    ])
            # This logic replaces the original loop to correctly handle segments based on sil_tags
            # The original loop was: 
            # for i in range(len(sil_tags) - 1):
            #    chunks.append([
            #        self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]),
            #        int(sil_tags[i][1] * self.hop_size),
            #        int(sil_tags[i + 1][0] * self.hop_size),
            #    ])
            # if sil_tags[-1][1] < total_frames_hops:
            #    chunks.append([
            #        self._apply_slice(waveform, sil_tags[-1][1], total_frames_hops),
            #        int(sil_tags[-1][1] * self.hop_size),
            #        int(total_frames_hops * self.hop_size),
            #    ])
        logger.info(f"Waveform sliced into {len(chunks)} chunks.")
        return chunks

def format_seconds_to_hms(seconds):
    # logger.debug(f"Formatting {seconds} seconds to HH:MM:SS")
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds_val = seconds % 60
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, int(seconds_val))


def create_spanish_metadata(name_project, ch_tokenizer):
    path_project = os.path.join(path_data, name_project)
    path_project_wavs = os.path.join(path_project, "wavs")
    file_metadata_csv = os.path.join(path_project, "metadata.csv") # Source CSV from transcribe_all
    file_arrow_output = os.path.join(path_project, "raw.arrow")
    file_duration_json = os.path.join(path_project, "duration.json")
    file_vocab_txt = os.path.join(path_project, "vocab.txt")

    if not os.path.isfile(file_metadata_csv):
        message = f"Metadata CSV file not found: {file_metadata_csv}. Run transcription first."
        print(message)
        return {"status": "error", "message": message, "new_vocal": ""}, ""

    with open(file_metadata_csv, "r", encoding="utf-8-sig") as f:
        data_csv = f.read()

    lines = data_csv.strip().split("\n")
    
    audio_path_list = []
    text_list = []
    duration_list = []
    result_for_arrow = []
    error_files_detail = []
    text_vocab_set = set()
    total_lines = len(lines)

    for idx, line in enumerate(lines):
        # logger.debug(f"Processing metadata line {idx+1}/{total_lines}: '{line}'") # Can be too verbose
        if (idx + 1) % 100 == 0:
             print(f"Processed {idx+1}/{total_lines} metadata lines...")
        sp_line = line.split("|")
        if len(sp_line) != 2:
            print(f"Skipping malformed line (expected 2 parts, got {len(sp_line)}): '{line}'")
            error_files_detail.append([line, "malformed line"])
            continue
        
        name_audio, text_raw = sp_line[:2]
        file_audio_full_path = os.path.join(path_project_wavs, f"{name_audio}.wav")

        if not os.path.isfile(file_audio_full_path):
            print(f"Audio file missing for segment '{name_audio}': {file_audio_full_path}")
            error_files_detail.append([file_audio_full_path, "missing audio file"])
            continue
        try:
            duration = get_audio_duration(file_audio_full_path)
        except Exception as e_dur:
            print(f"Could not get duration for '{file_audio_full_path}': {e_dur}")
            error_files_detail.append([file_audio_full_path, f"duration error: {e_dur}"])
            continue
        
        if not (1 <= duration <= 30):
            reason = "too long (>30s)" if duration > 30 else "too short (<1s)"
            print(f"Segment '{name_audio}' duration {duration:.2f}s is {reason}. Skipping.")
            error_files_detail.append([file_audio_full_path, reason])
            continue
        
        if len(text_raw.strip()) < 3:
            print(f"Segment '{name_audio}' text '{text_raw}' is too short. Skipping.")
            error_files_detail.append([file_audio_full_path, "text too short"])
            continue
        
        text_cleaned = clear_text(text_raw)
        try:
            # Assuming convert_char_to_pinyin expects a list and returns a list
            text_pinyin = convert_char_to_pinyin([text_cleaned], polyphone=True)[0]
        except Exception as e_pinyin:
            print(f"Error converting text to pinyin for '{text_cleaned}': {e_pinyin}")
            error_files_detail.append([file_audio_full_path, f"pinyin conversion error: {e_pinyin}"])
            continue

        audio_path_list.append(file_audio_full_path)
        duration_list.append(duration)
        text_list.append(text_pinyin)
        result_for_arrow.append({"audio_path": file_audio_full_path, "text": text_pinyin, "duration": duration})
        if ch_tokenizer:
            text_vocab_set.update(list(text_pinyin)) # Use pinyin for vocab if ch_tokenizer is true

    if not duration_list:
        message = f"No valid audio files/segments found after filtering in project: {name_project}"
        print(message)
        return {"status": "warning", "message": message, "new_vocal": ""}, ""

    min_second = round(min(duration_list), 2)
    max_second = round(max(duration_list), 2)
    total_duration_seconds = sum(duration_list)
    print(f"Total valid segments: {len(text_list)}, Total duration: {format_seconds_to_hms(total_duration_seconds)}")
    print(f"Min segment duration: {min_second}s, Max segment duration: {max_second}s")

    print(f"Writing {len(result_for_arrow)} entries to Arrow file: {file_arrow_output}")
    try:
        with ArrowWriter(path=file_arrow_output, writer_batch_size=1) as writer: # Increased batch size
            for record_idx, record_data in enumerate(result_for_arrow):
                writer.write(record_data)
                # if (record_idx + 1) % 500 == 0:
                #     logger.debug(f"Wrote {record_idx+1}/{len(result_for_arrow)} records to Arrow file...")
        print(f"Successfully wrote Arrow file: {file_arrow_output}")
    except Exception as e_arrow:
        message = f"Error writing Arrow file {file_arrow_output}: {e_arrow}"
        print(message, exc_info=True)
        return {"status": "error", "message": message, "new_vocal": ""}, ""

    print(f"Writing duration info to JSON: {file_duration_json}")
    with open(file_duration_json, "w") as f_json:
        json.dump({"duration": duration_list}, f_json, ensure_ascii=False, indent=2)
    print(f"Successfully wrote duration JSON: {file_duration_json}")

    new_vocal_content = ""
    vocab_size = 0
    if not ch_tokenizer:
        # Path to a pre-existing vocabulary file for pinyin (non-character based tokenizer)
        file_vocab_finetune_src = os.path.join(path_data, "Emilia_ZH_EN_pinyin/vocab.txt") # Example path
        logger.info(f"Using pre-existing vocabulary (ch_tokenizer=False) from: {file_vocab_finetune_src}")
        if not os.path.isfile(file_vocab_finetune_src):
            message = f"Base vocabulary file '{file_vocab_finetune_src}' not found for non-character tokenizer!"
            logger.error(message)
            return {"status": "error", "message": message, "new_vocal": ""}, ""
        shutil.copy2(file_vocab_finetune_src, file_vocab_txt)
        logger.info(f"Copied base vocabulary to: {file_vocab_txt}")
        with open(file_vocab_txt, "r", encoding="utf-8-sig") as vf:
            vocab_size = sum(1 for _ in vf)
    else:
        logger.info(f"Generating new vocabulary (ch_tokenizer=True) from processed texts.")
        sorted_vocab = sorted(list(text_vocab_set))
        with open(file_vocab_txt, "w", encoding="utf-8-sig") as f_vocab:
            for vocab_char in sorted_vocab:
                f_vocab.write(vocab_char + "\n")
                new_vocal_content += vocab_char + "\n"
        vocab_size = len(sorted_vocab)
        logger.info(f"Created new vocabulary file: {file_vocab_txt} with {vocab_size} entries.")

    error_text_summary = "\n".join([f"File: {item[0]}, Reason: {item[1]}" for item in error_files_detail]) if error_files_detail else "No errors."
    
    summary_message = (
        f"Metadata creation complete for project '{name_project}'.\n"
        f"Total valid samples processed: {len(text_list)}.\n"
        f"Total audio duration: {format_seconds_to_hms(total_duration_seconds)}.\n"
        f"Min segment duration: {min_second}s, Max segment duration: {max_second}s.\n"
        f"Arrow data file created at: {file_arrow_output}.\n"
        f"Vocabulary file created at: {file_vocab_txt} (Size: {vocab_size}).\n"
        f"Duration JSON created at: {file_duration_json}."
    )
    status_val = "success"
    if error_files_detail:
        summary_message += f"\n\nEncountered {len(error_files_detail)} issues during processing:\n{error_text_summary}"
        logger.warning(f"Metadata creation for '{name_project}' completed with {len(error_files_detail)} issues. Summary: {summary_message}")
        status_val = "warning"
    else:
        logger.info(f"Metadata creation for '{name_project}' successful. Summary: {summary_message}")

    return {
        "status": status_val,
        "message": summary_message,
        "samples_count": len(text_list),
        "total_duration_hms": format_seconds_to_hms(total_duration_seconds),
        "min_duration_s": min_second,
        "max_duration_s": max_second,
        "arrow_file_path": file_arrow_output,
        "vocab_file_path": file_vocab_txt,
        "vocab_size": vocab_size,
        "duration_json_path": file_duration_json,
        "error_details_count": len(error_files_detail),
        "error_details_summary": error_text_summary
    }, new_vocal_content


def spanish_transcribe_all(name_project, audio_files, language, user_mode=False):
    logger.info(f"Starting transcription process for project '{name_project}', language '{language}', user_mode={user_mode}")
    path_project = os.path.join(path_data, name_project)
    path_dataset = os.path.join(path_project, "dataset")
    path_project_wavs = os.path.join(path_project, "wavs")
    file_metadata = os.path.join(path_project, "metadata.csv")

    if not user_mode and not audio_files:
        logger.error("No audio files provided for transcription when not in user_mode.")
        return {"status": "error", "message": "You need to load an audio file.", "error_count": 0}

    if os.path.isdir(path_project_wavs):
        logger.warning(f"'{path_project_wavs}' already exists, removing it.")
        shutil.rmtree(path_project_wavs)
    if os.path.isfile(file_metadata):
        logger.warning(f"'{file_metadata}' already exists, removing it.")
        os.remove(file_metadata)
    os.makedirs(path_project_wavs, exist_ok=True)
    logger.info(f"Ensured wavs directory exists: {path_project_wavs}")

    files_to_process = []
    if user_mode:
        logger.debug(f"User mode: searching for audio files in {path_dataset}")
        for fmt in ("*.wav", "*.ogg", "*.opus", "*.mp3", "*.flac"):
            files_to_process.extend(glob(os.path.join(path_dataset, fmt)))
        if not files_to_process:
            logger.error(f"No audio files found in dataset directory: {path_dataset}")
            return {"status": "error", "message": "No audio file was found in the dataset.", "error_count": 0}
        logger.info(f"Found {len(files_to_process)} audio files in dataset for user_mode.")
    else:
        files_to_process = audio_files
        logger.info(f"Processing {len(files_to_process)} provided audio files.")

    alpha = 0.5
    _max = 1.0
    slicer = Slicer(sr=24000) # Assuming 24kHz for F5TTS
    num_segments = 0
    error_num = 0
    metadata_content = ""
    total_files = len(files_to_process)
    logger.info(f"Starting slicing and transcription for {total_files} audio files.")

    for i, file_audio_path in enumerate(files_to_process):
        logger.debug(f"Processing file {i+1}/{total_files}: {file_audio_path}")
        try:
            audio, sr = librosa.load(file_audio_path, sr=24000, mono=True)
            logger.debug(f"Loaded audio: {file_audio_path}, duration: {len(audio)/sr:.2f}s, sr: {sr}")
            list_slicer_chunks = slicer.slice(audio)
            logger.info(f"File '{os.path.basename(file_audio_path)}' sliced into {len(list_slicer_chunks)} chunks.")

            for chunk_idx, (chunk_audio, start_time, end_time) in enumerate(list_slicer_chunks):
                segment_name = f"segment_{num_segments}"
                file_segment_path = os.path.join(path_project_wavs, f"{segment_name}.wav")
                # logger.debug(f"Processing chunk {chunk_idx+1}/{len(list_slicer_chunks)} for {file_audio_path}, saving to {file_segment_path}")
                
                tmp_max = np.abs(chunk_audio).max()
                if tmp_max > 1.0:
                    logger.warning(f"Segment {segment_name} max amplitude {tmp_max} > 1.0. Normalizing.")
                    chunk_audio /= tmp_max
                
                # The following normalization seems specific, keeping it as is.
                # chunk_audio = (chunk_audio / tmp_max * (_max * alpha)) + (1 - alpha) * chunk_audio 
                # This formula might be problematic if tmp_max is 0 or very small. Let's adjust to avoid division by zero if chunk is silent.
                if tmp_max > 1e-6: # Avoid division by zero/small numbers for silent chunks
                     chunk_normalized = (chunk_audio / tmp_max * (_max * alpha)) + (1 - alpha) * chunk_audio
                else:
                     chunk_normalized = chunk_audio # Keep as is if silent or near silent

                wavfile.write(file_segment_path, 24000, (chunk_normalized * 32767).astype(np.int16))
                # logger.debug(f"Saved segment {file_segment_path}")
                try:
                    text = spanish_transcribe(file_segment_path, language)
                    text = text.lower().strip().replace('"', "")
                    logger.debug(f"Transcription for {segment_name}: '{text}'")
                    metadata_content += f"{segment_name}|{text}\n"
                    num_segments += 1
                except Exception as e_transcribe:
                    logger.error(f"Transcription error for segment {file_segment_path}: {e_transcribe}", exc_info=True)
                    error_num += 1
        except Exception as e_load_slice:
            logger.error(f"Error loading or slicing file {file_audio_path}: {e_load_slice}", exc_info=True)
            error_num += 1 # Count this as an error for the main file

    logger.info(f"Writing metadata to {file_metadata}")
    with open(file_metadata, "w", encoding="utf-8-sig") as f:
        f.write(metadata_content)

    status_message = f"Transcription process complete. Total segments created: {num_segments}. Output WAVs in: {path_project_wavs}. Metadata: {file_metadata}."
    if error_num > 0:
        status_message += f" Encountered {error_num} errors during processing."
        logger.warning(status_message)
        status_val = "warning"
    else:
        logger.info(status_message)
        status_val = "success"
    
    return {
        "status": status_val,
        "message": status_message,
        "segments_created": num_segments,
        "output_wav_path": path_project_wavs,
        "metadata_file_path": file_metadata,
        "error_count": error_num
    }

from importlib.resources import files
import json
import os
import subprocess
path_data = str(files("src").joinpath("./Spanish_data"))
path_project_ckpts = str(files("src").joinpath("./Spanish_ckpts"))
file_train = str(files("src.Spanish_f5tts.Spanish_train").joinpath("finetune_cli.py"))
exp_name ="F5TTS_v1_Base"
def load_settings(project_name):
    project_name = project_name
    path_project = os.path.join(path_project_ckpts, project_name)
    file_setting = os.path.join(path_project, "setting.json")
    default_settings = {
        "exp_name": "F5TTS_v1_Base",
        "learning_rate": 1e-5,
        "batch_size_per_gpu": 3200,
        "batch_size_type": "frame",
        "max_samples": 19,
        "grad_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "epochs": 100,
        "num_warmup_updates": 20000,
        "save_per_updates": 1000,
        "keep_last_n_checkpoints": -1,
        "last_per_updates": 100,
        "finetune": True,
        "file_checkpoint_train": str(files("src").joinpath("./weights/pruned_model.safetensors")),
        "tokenizer_type": "pinyin",
        "tokenizer_file": "",
        "mixed_precision": "fp16",
        "logger": "none",
        "ch_8bit_adam": False,
    }
    if os.path.isfile(file_setting):
        with open(file_setting, "r") as f:
            file_settings = json.load(f)
        default_settings.update(file_settings)
    return default_settings


def save_settings(project_name, *args):
    (
        exp_name, learning_rate, batch_size_per_gpu, batch_size_type, max_samples,
        grad_accumulation_steps, max_grad_norm, epochs, num_warmup_updates,
        save_per_updates, keep_last_n_checkpoints, last_per_updates, finetune,
        file_checkpoint_train, tokenizer_type, tokenizer_file, mixed_precision,
        logger, ch_8bit_adam
    ) = args

    path_project = os.path.join(path_project_ckpts, project_name)
    os.makedirs(path_project, exist_ok=True)
    file_setting = os.path.join(path_project, "setting.json")

    settings = {
        "exp_name": exp_name,
        "learning_rate": learning_rate,
        "batch_size_per_gpu": batch_size_per_gpu,
        "batch_size_type": batch_size_type,
        "max_samples": max_samples,
        "grad_accumulation_steps": grad_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "epochs": epochs,
        "num_warmup_updates": num_warmup_updates,
        "save_per_updates": save_per_updates,
        "keep_last_n_checkpoints": keep_last_n_checkpoints,
        "last_per_updates": last_per_updates,
        "finetune": finetune,
        "file_checkpoint_train": file_checkpoint_train,
        "tokenizer_type": tokenizer_type,
        "tokenizer_file": tokenizer_file,
        "mixed_precision": mixed_precision,
        "logger": logger,
        "bnb_optimizer": ch_8bit_adam,
    }

    with open(file_setting, "w") as f:
        json.dump(settings, f, indent=4)
   
    return "Settings saved!"

def spanish_run_training(project_name):
    global training_process, tts_api, stop_signal
    settings = load_settings(project_name)
    args = [
        settings["exp_name"],
        settings["learning_rate"],
        settings["batch_size_per_gpu"],
        settings["batch_size_type"],
        settings["max_samples"],
        settings["grad_accumulation_steps"],
        settings["max_grad_norm"],
        settings["epochs"],
        settings["num_warmup_updates"],
        settings["save_per_updates"],
        settings["keep_last_n_checkpoints"],
        settings["last_per_updates"],
        settings["finetune"],
        settings["file_checkpoint_train"],
        settings["tokenizer_type"],
        settings["tokenizer_file"],
        settings["mixed_precision"],
        settings["logger"],
        settings["ch_8bit_adam"],
    ]

    save_settings(project_name, *args)

    tokenizer_type = "pinyin"

    if settings["mixed_precision"] != "none":
        fp16 = f"--mixed_precision={settings['mixed_precision']}"
    else:
        fp16 = ""

    project_base = project_name
    cmd = (
        f"PYTHONHASHSEED=random accelerate launch {fp16} {file_train} --exp_name {exp_name}"
        f" --learning_rate {settings['learning_rate']}"
        f" --batch_size_per_gpu {settings['batch_size_per_gpu']}"
        f" --batch_size_type {settings['batch_size_type']}"
        f" --max_samples {settings['max_samples']}"
        f" --grad_accumulation_steps {settings['grad_accumulation_steps']}"
        f" --max_grad_norm {settings['max_grad_norm']}"
        f" --epochs {settings['epochs']}"
        f" --num_warmup_updates {settings['num_warmup_updates']}"
        f" --save_per_updates {settings['save_per_updates']}"
        f" --keep_last_n_checkpoints {settings['keep_last_n_checkpoints']}"
        f" --last_per_updates {settings['last_per_updates']}"
        f" --dataset_name {project_base}"
    )

    if settings["finetune"]:
        cmd += " --finetune"
    if settings["file_checkpoint_train"]:
        cmd += f" --pretrain {settings['file_checkpoint_train']}"
    if settings["tokenizer_file"]:
        cmd += f" --tokenizer_path {settings['tokenizer_file']}"
    cmd += f" --tokenizer {tokenizer_type}"
    if settings["logger"] != "none":
        cmd += f" --logger {settings['logger']}"
    cmd += " --log_samples"
    if settings["ch_8bit_adam"]:
        cmd += " --bnb_optimizer"
        
    project_name = project_name.replace("_pinyin", "")
    settings["tokenizer_file"] = os.path.join(path_data, project_name, "vocab.txt")

    print(f"Launching training command:\n{cmd}")
    #yield(f"Launching training command:\n{cmd}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    training_process = subprocess.Popen(cmd, shell=True, env=env)
    yield "Training started please wait"
    training_process.wait()
    yield "Training Completed"
    #return "Training started."
# --- Flask API Endpoints ---

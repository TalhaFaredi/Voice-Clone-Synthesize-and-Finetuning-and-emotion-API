import os 
import logging
from importlib.resources import files

try:
    path_data = str(files("src").joinpath("./English_data"))
    path_project_ckpts = str(files("src").joinpath("./English_ckpts"))
    file_train = str(files("src.English_f5tts.English_train").joinpath("finetune_cli.py"))

except Exception as e:
    path_data = "./src/English_data"
    path_project_ckpts = "./src/English_ckpts"
    os.makedirs(path_data, exist_ok=True)
    os.makedirs(path_project_ckpts, exist_ok=True)

def get_list_projects():
    project_list = []
    for folder in os.listdir(path_data):
        path_folder = os.path.join(path_data, folder)
        if not os.path.isdir(path_folder):
            continue
        folder = folder.lower()
        if folder == "emilia_zh_en_pinyin":
            continue
        project_list.append(folder)
    projects_selelect = None if not project_list else project_list[-1]
    return project_list, projects_selelect


import os 
import logging
from importlib.resources import files

try:
    path_data = str(files("src").joinpath("./Spanish_data"))
    path_project_ckpts = str(files("src").joinpath("./Spanish_ckpts"))
    file_train = str(files("src.Spanish_f5tts.Spanish_train").joinpath("finetune_cli.py"))

except Exception as e:
    path_data = "./src/Spanish_data"
    path_project_ckpts = "./src/Spanish_ckpts"
    os.makedirs(path_data, exist_ok=True)
    os.makedirs(path_project_ckpts, exist_ok=True)




def create_spanish_data_project(name, tokenizer_type):
   
    project_name_with_type = name + "_" + tokenizer_type
    project_path = os.path.join(path_data, project_name_with_type)
    dataset_path = os.path.join(project_path, "dataset")
    try:
        os.makedirs(project_path, exist_ok=True)

        os.makedirs(dataset_path, exist_ok=True)
  
        return project_name_with_type
    except OSError as e:
        print(f"OSError creating project directories for '{project_name_with_type}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error creating project directories for '{project_name_with_type}': {e}")
        return None

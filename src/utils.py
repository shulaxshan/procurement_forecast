import os
import sys
import numpy as np 
import pandas as pd
import pickle
from src.exception import CustomException



def save_object(file_path, unique_ids_):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file:
            for unique_id in unique_ids_:
                file.write(f"{unique_id}\n".encode('latin1'))

    except Exception as e:
        raise CustomException(e,sys)
    

def save_variable(file_path, my_variable):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(my_variable, file)

    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

# def read_yaml(path_to_yaml_path):
#     try:
#         with open(path_to_yaml_path) as yaml_file:
#             content = yaml.safe_load(yaml_file)
#             return content
        
#     except Exception as e:
#         raise CustomException(e, sys)
    

# def read_db_config(config_file_path):
#     try:
#         with open(config_file_path, 'r') as file:
#             config = yaml.safe_load(file)
#         db_config = config['db']
#         db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
#         return db_url
    
#     except Exception as e:
#         raise CustomException(e, sys)



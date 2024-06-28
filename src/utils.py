import os
import sys
import numpy as np 
import pandas as pd
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
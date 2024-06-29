import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

STAGE_NAME = "Data Transformation stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        final_data_path = 'artifacts/data_ingestion/final_data.csv'  # Replace with the actual path to your final data file
        data_transform = DataTransformation()
        data_transform.data_preprocessor(final_data_path)


if __name__ == '__main__':
    try:
        logging.info(f">>>>>>>>> stage {STAGE_NAME} started  <<<<<<<<<<")
        obj = DataTransformationPipeline
        obj.main()
        logging.info(f">>>>>>>>> stage {STAGE_NAME} completed  <<<<<<<<<<")

    except CustomException as e:
        raise CustomException(e,sys)
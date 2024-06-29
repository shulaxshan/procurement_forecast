import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()



if __name__ == '__main__':
    try:
        logging.info(f">>>>>>>>> stage {STAGE_NAME} started  <<<<<<<<<<")
        obj = DataIngestionPipeline()
        # final_data_path, category_data_path = obj.main()
        final_data, category_data = obj.main()
        logging.info(f">>>>>>>>> stage {STAGE_NAME} completed  <<<<<<<<<<")

    except CustomException as e:
        raise CustomException(e,sys)
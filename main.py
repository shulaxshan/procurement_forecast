import sys
from src.logger import logging
from src.exception import CustomException
from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.pipeline.stage_02_data_transformation import DataTransformationPipeline


STAGE_NAME = "Data Ingestion stage"
try:
    logging.info(f">>>>>>>>> stage {STAGE_NAME} started  <<<<<<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logging.info(f">>>>>>>>> stage {STAGE_NAME} completed  <<<<<<<<<<")

except CustomException as e:
    raise CustomException(e,sys)



STAGE_NAME = "Data Transformation stage"
try:
    logging.info(f">>>>>>>>> stage {STAGE_NAME} started  <<<<<<<<<<")
    data_transformation = DataTransformationPipeline()
    data_transformation.main()
    logging.info(f">>>>>>>>> stage {STAGE_NAME} completed  <<<<<<<<<<")

except CustomException as e:
    raise CustomException(e,sys)



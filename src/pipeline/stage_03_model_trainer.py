import sys
from src.exception import CustomException
from src.logger import logging
from src.components.model_trainer import ModelTraining
from src.utils import load_object

STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        df_path = 'artifacts/data_transformation/weekly_data.csv'  # Replace with the actual path to your final data file
        max_data_path = 'artifacts/data_transformation/my_max_date_variable.pkl'
        max_date = load_object(max_data_path)
        logging.info(f'Successfully taken the max date from pickle file. Max date is: {max_date}')
        model_train = ModelTraining()
        model_train.initiate_model_forecast(df_path,max_date)


if __name__ == '__main__':
    try:
        logging.info(f">>>>>>>>> stage {STAGE_NAME} started  <<<<<<<<<<")
        obj = ModelTrainingPipeline
        obj.main()
        logging.info(f">>>>>>>>> stage {STAGE_NAME} completed  <<<<<<<<<<")

    except CustomException as e:
        raise CustomException(e,sys)

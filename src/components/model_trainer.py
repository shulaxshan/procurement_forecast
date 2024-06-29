import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from prophet import Prophet
from math import ceil
import inspect
from src.utils import save_object,save_variable


@dataclass
class ModelTrainingConfig:
    forecasted_df_file_paths = os.path.join('artifacts/model_trainer', "forecasted_df.csv")
    actual_df_file_paths = os.path.join('artifacts/model_trainer', "actual_df.csv")
    failded_ids_path = os.path.join('artifacts/model_trainer',"failed_unique_ids_supplier_category.txt")


class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()


    def replace_negatives_with_weighted_average(self,forecast_data):

        replaced_count = 0
        updated_forecast = forecast_data.copy()
        for i, val in enumerate(forecast_data):
            if val < 0:  # Handle negative values
                closest_non_negative_1 = None
                closest_non_negative_2 = None
                distance_1 = None
                distance_2 = None
                # Find the two closest non-negative numbers
                for j in range(1, min(i+1, len(forecast_data)-i)):
                    if forecast_data[i-j] >= 0:
                        closest_non_negative_1 = forecast_data[i-j]
                        distance_1 = j
                    if forecast_data[i+j] >= 0:
                        closest_non_negative_2 = forecast_data[i+j]
                        distance_2 = j
                    if closest_non_negative_1 is not None and closest_non_negative_2 is not None:
                        break
                # Compute the weights based on the inverse of the distances
                if closest_non_negative_1 is not None and closest_non_negative_2 is not None:
                    weight_1 = 1 / (distance_1 + 1)  # Adding 1 to avoid division by zero
                    weight_2 = 1 / (distance_2 + 1)  # Adding 1 to avoid division by zero
                    total_weight = weight_1 + weight_2
                    weighted_avg = (closest_non_negative_1 * weight_1 + closest_non_negative_2 * weight_2) / total_weight
                    updated_forecast[i] = weighted_avg
                replaced_count+=1 
        return updated_forecast, replaced_count


    def model_(self,df,max_date):
        try:
            # logging.info("Initiating data transformation")

            # df = pd.read_csv(preprocessed_file_path,encoding=encoding)

            def week_of_month(dt):
                    first_day = dt.replace(day=1)
                    dom = dt.day
                    adjusted_dom = dom + first_day.weekday()
                    return int(ceil(adjusted_dom/7.0))

            
            # print(df.Booking_Date.max())
            df['Week_Number'] = df['Booking_Date'].apply(lambda x: week_of_month(x))
            prophet_df = df.rename(columns={'Booking_Date': 'ds', 'Gross_Amount': 'y'})

            model = Prophet(interval_width = 0.8)
            model.fit(prophet_df)
            future = model.make_future_dataframe(freq='W',periods=52)
            forecast = model.predict(future)
            forecast_sel_col = ['ds', 'yhat']
            forecast = forecast[forecast_sel_col]
            forecast.rename(columns={'ds': 'Booking_Date', 'yhat':'predicted'}, inplace=True)

            logging.info("Prophet model forecasting completed successfully")
            
            ## Handling the negative values with a specific function
            forecast['predicted'], _ = self.replace_negatives_with_weighted_average(forecast['predicted'])
            forecast['uniqueID'] = df['uniqueID'].iloc[-1]
            merged_df = pd.merge(df, forecast, on=['uniqueID','Booking_Date'], how ='right')
            merged_df_ = merged_df[['uniqueID','Booking_Date', 'Gross_Amount', 'predicted']]


            ##### weekly Predicted value aggregated into monthly
            prophet_agg_prediction_df = merged_df_.groupby(['uniqueID', pd.Grouper(key='Booking_Date', freq='ME')]).sum().reset_index()
            #### Predicted value if negative it will be convert it into zero
            prophet_agg_prediction_df['predicted'] = prophet_agg_prediction_df['predicted'].apply(lambda x: max(0,x))

            #### Actual Data frame
            actual_df = prophet_agg_prediction_df[prophet_agg_prediction_df['Booking_Date']<= max_date]
            sel_col = ['uniqueID','Booking_Date','Gross_Amount']
            actual_df = actual_df[sel_col]

            #### Prediction Data frame
            prediction_df = prophet_agg_prediction_df[prophet_agg_prediction_df['Booking_Date']> max_date]
            sel_col_pre = ['uniqueID','Booking_Date','predicted']
            prediction_df = prediction_df[sel_col_pre]

            return prediction_df, actual_df
        
        except CustomException as e:
            raise CustomException(e,sys)

    def run_model_for_all_ids(self, model, filtered_df, max_date=None,model_name=""):
        try:
            # Initialize empty DataFrames to store the results
            all_prediction_df = pd.DataFrame()
            all_actual_df = pd.DataFrame()

            # List to store unique IDs that fail
            failed_unique_ids = []

            # Check if the model function requires max_date as an argument
            model_args = inspect.signature(model).parameters

            # Iterate over unique IDs for Prophet model
            for unique_id in filtered_df['uniqueID'].unique():
                try:
                    logging.info(f"Processing unique ID: {unique_id}")
                    # Filter dataframe for each unique ID
                    df_subset = filtered_df[filtered_df['uniqueID'] == unique_id].reset_index(drop=True)
                    logging.info(df_subset.head())
                    
                    # Apply model function to the subset of data
                    if 'max_date' in model_args:
                        prediction_df, actual_df = model(df_subset, max_date)
                    else:
                        prediction_df, actual_df = model(df_subset)
                    
                    logging.info(f"Prediction DataFrame for {unique_id}: {prediction_df.head()}")
                    logging.info(f"Actual DataFrame for {unique_id}: {actual_df.head()}")
                    
                    # Append the results to the main result DataFrames
                    all_prediction_df = pd.concat([all_prediction_df,prediction_df], ignore_index=True)
                    all_actual_df = pd.concat([all_actual_df,actual_df], ignore_index=True)

                except Exception as e:
                    logging.error(f"Failed to process unique ID {unique_id}: {e}")
                    failed_unique_ids.append(unique_id)

            # Write failing unique IDs to a text file
            if failed_unique_ids:
                 logging.info("Writing failed unique IDs to failed_unique_ids_item_level.txt")
            #     # Check if file exists, and open in append mode if it does
                 save_object(
                        file_path =self.model_training_config.failded_ids_path,
                        unique_ids_ = failed_unique_ids
                        )
            #     logger.debug(f"Failed unique IDs written to failed_unique_ids_item_level.txt")

            logging.info("Successfully saved both actual data and prophet prediction results as two separate dataframes.")
            return all_prediction_df, all_actual_df
        
        except CustomException as e:
            raise CustomException(e,sys)
        
    
    def initiate_model_forecast(self,df_path,max_date):
        try:
            logging.info("Initiating Model Training")
            df = pd.read_csv(df_path)
            df['Booking_Date'] = pd.to_datetime(df['Booking_Date'])
            df['Gross_Amount'] = df['Gross_Amount'].astype('float64')
            print(df.dtypes)
            print("model df columns",df.columns)

            prop1_all_prediction_df, prop1_all_actual_df = self.run_model_for_all_ids(self.model_,df,max_date,model_name="Prophet_model")

            os.makedirs(os.path.dirname(self.model_training_config.forecasted_df_file_paths),exist_ok=True)
            prop1_all_prediction_df.to_csv(self.model_training_config.forecasted_df_file_paths,index=False,header=True,encoding='latin1')
            prop1_all_actual_df.to_csv(self.model_training_config.actual_df_file_paths,index=False,header=True,encoding='latin1')

            logging.info(f'Successfully saved predictions into csv file.Predicted unique_IDS count: {prop1_all_prediction_df['uniqueID'].nunique()}')
   
            return prop1_all_prediction_df, prop1_all_actual_df
        except CustomException as e:
            raise CustomException(e,sys)
    


        






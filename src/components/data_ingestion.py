import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTraining
from src.components.model_trainer import ModelTrainingConfig



@dataclass
class DataIngestionConfig:
    final_data_path: str=os.path.join('artifacts/data_ingestion',"final_data.csv")
    category_data_path: str=os.path.join('artifacts/data_ingestion',"category_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:

            category_df = pd.read_csv('notebook/data/category_with_ID.csv',encoding = 'latin1')
            category_df = category_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            category_df['CategoryID'] = category_df['CategoryID'].astype(str)
            print('Category data shape',category_df.shape)

            property_data = pd.read_csv('notebook/data/filtered_with_feb_with_stand_name.csv',encoding = 'latin1', 
                        usecols=['Property name','Booking Date','Gross Amount','Category','CategoryID','Cost Center Name','Item name'])
            property_data.rename(columns={'Property name':'Property_name','Booking Date':'Booking_Date','Gross Amount':'Gross_Amount',
                                        'Category':'Category','CategoryID':'CategoryID','Cost Center Name':'Cost_Center_Name','Item name':'Item_Name'}, inplace=True)
            
            # Remove leading and trailing whitespaces from all values in all columns
            property_data = property_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            property_data['Item_Name'] = property_data['Item_Name'].str.lower()
            print("Procurement data shape:",property_data.shape)

            mapping_df = pd.read_csv('notebook/data/item_name_mapping.csv',encoding = 'latin1'
                                        ,usecols=['Item_Name','Item_Name_Standard','Category_Standard','Remove_flag','CategoryID_Standard'])

            mapping_df = mapping_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            # mapping_df['Category_Standard'] = mapping_df['Category_Standard'].str.lower()
            mapping_df['Item_Name'] = mapping_df['Item_Name'].str.lower()
            mapping_df['Item_Name_Standard'] = mapping_df['Item_Name_Standard'].str.lower()
            mapping_df_ = mapping_df.drop_duplicates().reset_index(drop=True)
            mapping_df_['Remove_flag'] = mapping_df_['Remove_flag'].astype(str)
            mapping_df_ = mapping_df_[mapping_df_['Remove_flag']=='0']
            print('Mapping data shape',mapping_df_.shape)
            
            merged_property_data_df = pd.merge(property_data, mapping_df_[['Item_Name','Category_Standard','Item_Name_Standard','CategoryID_Standard']], on=['Item_Name'], how='inner')
            merged_property_data_df['CategoryID_Standard'] = merged_property_data_df['CategoryID_Standard'].astype(str)
            #### Get only 26 categories data
            final_data_set = merged_property_data_df[merged_property_data_df['CategoryID_Standard'].isin(category_df.CategoryID.tolist())].reset_index(drop=True)
            print('Final data shape',final_data_set.shape)

            logging.info('Read all dataset and prepared final data as dataframe to preform data tranformation')

            os.makedirs(os.path.dirname(self.ingestion_config.final_data_path),exist_ok=True)

            final_data_set.to_csv(self.ingestion_config.final_data_path,index=False,header=True,encoding='latin1')
            category_df.to_csv(self.ingestion_config.category_data_path,index=False,header=True,encoding='latin1')

            logging.info("Data ingestion completed")
            
            # return (self.ingestion_config.final_data_path,self.ingestion_config.category_data_path)
            return final_data_set,category_df
        
        except CustomException as e:
            logging.error(e)


# if __name__ == "__main__":
#     data_ingestion=DataIngestion()
#     final_data_path, category_data_path = data_ingestion.initiate_data_ingestion()

#     data_transform = DataTransformation()
#     selected_df_daily,selected_df_weekly,selected_df_monthly, max_date = data_transform.data_preprocessor(final_data_path)

#     model_trainer =ModelTraining()
#     model_trainer.initiate_model_forecast(selected_df_weekly,max_date)


import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import save_object
from src.logger import logging
import os

import numpy as np
import pandas as pd
from math import ceil
import calendar


@dataclass
class DataTransformationConfig:
    # preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    selected_uniqueIDs_path = os.path.join('artifacts', "selected_unique_ids.txt")
    unselected_uniqueIDs_path = os.path.join('artifacts', "unselected_unique_ids.txt")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def fill_missing_dates(self,df, unique_id_col, date_col,max_date_):
        try:
            # Convert 'Booking_Date' column to datetime if it's not already
            df[date_col] = pd.to_datetime(df[date_col])

            date_range = pd.date_range(start=df.Booking_Date.min(), end= max_date_+pd.offsets.MonthEnd(0), freq='D')

            # Create a MultiIndex with all unique IDs and the generated date range
            idx = pd.MultiIndex.from_product([df[unique_id_col].unique(), date_range], names=[unique_id_col, date_col])

            # Reindex the DataFrame with the MultiIndex and fill missing values with zeros
            filled_df = df.set_index([unique_id_col, date_col]).reindex(idx, fill_value=0).reset_index()

            return filled_df
        except CustomException as e:
            raise CustomException(e,sys)
        

    def get_days_week_month_count(self,df):
        try:
            # Add a new column 'Year'
            df['Year'] = df['Booking_Date'].dt.isocalendar().year
            # Add a new column 'Week of Year'
            df['Week_of_Year'] = df['Booking_Date'].dt.isocalendar().week

            df['week_year'] = df['Year'].astype(str) + '-' + df['Week_of_Year'].astype(str)

            # Group the original dataframe by 'Unique ID'
            grouped_df = df.groupby(['uniqueID']).agg(
                no_of_days=('Booking_Date', 'nunique'),  # Count the number of unique booking dates
                no_of_weeks=('week_year', 'nunique'),
                no_of_months=('Booking_Date', lambda x: x.dt.to_period('M').nunique()),
                last_Trx_Date = ('Booking_Date', 'max')).reset_index()
            
            grouped_df['lastest_date'] = grouped_df.last_Trx_Date.max()
            grouped_df['months_difference'] = (grouped_df['lastest_date'].dt.to_period('M') - grouped_df['last_Trx_Date'].dt.to_period('M'))
            grouped_df['months_difference'] = grouped_df['months_difference'].apply(lambda x: x.n)
            
            #### Filter uniqueIDs which has more week numbers
            grouped_df_more_week_df = grouped_df[~((grouped_df['no_of_days'] < 11) | (grouped_df['months_difference'] >= 4))]
            more_week_count_list_ = grouped_df_more_week_df.uniqueID.tolist()
            print('Selected Unique IDs count: ',len(more_week_count_list_))
            # print('Selected Unique IDs: ',more_week_count_list_)


            #### Filter uniqueIDs which has less week numbers
            grouped_df_less_week_df = grouped_df[(grouped_df['no_of_days']< 11) | (grouped_df['months_difference']>= 4)]  #### filter less than or equal 10 records and last 4 month does not happend any purchase
            less_count_unique_ids_list = grouped_df_less_week_df.uniqueID.tolist()

            print('Unselected Unique IDs count: ',len(less_count_unique_ids_list))
            # print('Unselected Unique IDs: ',less_count_unique_ids_list)

            return more_week_count_list_, less_count_unique_ids_list
        
        except CustomException as e:
            raise CustomException(e,sys)
        

    def weekly_resampling(self,df):
        try:
            sunday_dates = []
            def week_of_month(dt):
                first_day = dt.replace(day=1)
                dom = dt.day
                adjusted_dom = dom + first_day.weekday()
                return int(ceil(adjusted_dom/7.0))
            
            def calculate_sunday_date(year, month, week_number):
                first_day_of_month = pd.Timestamp(year, month, 1)
                first_day_of_month_weekday = first_day_of_month.dayofweek
                days_to_sunday = 6 - first_day_of_month_weekday
                first_sunday_of_month = first_day_of_month + pd.Timedelta(days=days_to_sunday)
                sunday_date = first_sunday_of_month + pd.Timedelta(weeks=week_number - 1)
                return sunday_date
            
            df['Year'] = df['Booking_Date'].dt.year
            df['Month'] = df['Booking_Date'].dt.month
            df['Week_Number'] = df['Booking_Date'].apply(lambda x: week_of_month(x))
            df_weekly = df.groupby(['uniqueID', 'Year', 'Month', 'Week_Number']).agg({'Gross_Amount': 'sum'}).reset_index()
            grouped = df_weekly.groupby(['uniqueID', 'Year', 'Month'])
            for name, group in grouped:
                last_two_weeks = group.tail(2)
                last_day_of_month = calendar.monthrange(name[1], name[2])[1]
                weekday_of_last_day = calendar.weekday(name[1], name[2], last_day_of_month)
        
                if weekday_of_last_day == 6:
                    continue
                else:
                    sum_last_two_weeks = last_two_weeks['Gross_Amount'].sum()
                    min_week_number = min(last_two_weeks['Week_Number'])
                    df_weekly.loc[last_two_weeks.index, 'Gross_Amount'] = sum_last_two_weeks
                    df_weekly.loc[last_two_weeks.index, 'Week_Number'] = min_week_number
            df_weekly = df_weekly.drop_duplicates()
            
            for index, row in df_weekly.iterrows():
                sunday_date = calculate_sunday_date(row['Year'], row['Month'], row['Week_Number'])
                sunday_dates.append(sunday_date)
        
            df_weekly['Sunday_Date'] = sunday_dates
            columns_to_select = ['uniqueID', 'Sunday_Date', 'Gross_Amount']
            df_weekly = df_weekly[columns_to_select].copy()
            df_weekly = df_weekly.rename(columns={'Sunday_Date':'Booking_Date'}) 

            #### Select wanted columns only 
            selected_colm = ['uniqueID','Booking_Date','Gross_Amount'] 
            df_weekly = df_weekly[selected_colm]

            return df_weekly
        
        except CustomException as e:
            raise CustomException(e,sys)
        


    def data_preprocessor(self,final_data_path):
        try:
            logging.info("Initiating data transformation")

            df = pd.read_csv(final_data_path)

            df['Booking_Date'] = pd.to_datetime(df['Booking_Date'])
            max_date = df.Booking_Date.max() +  pd.offsets.MonthEnd(0)
            print('Maximum Date: ',max_date)
            
            ### Create the uniqueID
            separator = '__'
            columns_to_concat = ['Property_name', 'Cost_Center_Name','Category_Standard', 'Item_Name_Standard']
            df['uniqueID'] = df[columns_to_concat].astype(str).apply(lambda x: x.str.cat(sep=separator), axis=1)
            logging.info("Sucessfully created UniqueID")

            #### Sort the data frame
            df = df.sort_values(by=['uniqueID','Booking_Date'])

            #### select only specfic column 
            select_columns = ['Booking_Date','uniqueID','Gross_Amount']
            selected_colm_df = df[select_columns]
            
            #### Groping the dataset by 'uniqueID','Booking_Date'
            selected_grouped_df = selected_colm_df.groupby(['uniqueID','Booking_Date'])['Gross_Amount'].sum().reset_index()
            logging.info("Successfully aggregated same-day transactions by unique ID")

            #### Filter only have more week numbers
            selected_grouped_df1 = selected_grouped_df.copy()
            more_week_count_list_, less_week_count_list_ = self.get_days_week_month_count(selected_grouped_df1)
            selected_filtered_df = selected_grouped_df[selected_grouped_df['uniqueID'].isin(more_week_count_list_)].reset_index(drop=True)

            sel_unique_ids = selected_filtered_df['uniqueID'].unique()
            print('Selected final uniqueIDs to perform forecast: ',len(sel_unique_ids))
            logging.info(f"Successfully filtered the uniqueIDs if have more week numbers. Selected unique IDs count: {len(sel_unique_ids)}")

        
            ### Write unselected unique IDs to a text file
            save_object(
                        file_path =self.data_transformation_config.selected_uniqueIDs_path,
                        unique_ids_ = sel_unique_ids
                        )
 
            #### Filter have less week numbers
            unselected_filtered_df = selected_grouped_df[selected_grouped_df['uniqueID'].isin(less_week_count_list_)].reset_index(drop=True)
            unsel_unique_ids = unselected_filtered_df['uniqueID'].unique()
            print('Unselected uniqueIDs counts: ',len(unsel_unique_ids))
            logging.info(f"Successfully filtered the uniqueIDs if have fewer days. Unselected unqiue IDs count: {len(unsel_unique_ids)}")

            save_object(
                        file_path =self.data_transformation_config.unselected_uniqueIDs_path,
                        unique_ids_ = unsel_unique_ids
                        )

            def sub_pre(dff):
                ### Fill missing sequence with zero
                try:
                    filled_data = []
                    for unique_id in dff['uniqueID'].unique():
                        subset = dff[dff['uniqueID'] == unique_id]  ### Get each uniqueIDs in loop
                        filled_subset = self.fill_missing_dates(subset, unique_id_col='uniqueID', date_col='Booking_Date', max_date_ = max_date)
                        filled_data.append(filled_subset)
                    
                    filled_missing_df = pd.concat(filled_data) # Concatenate the filled sequence subsets
                    logging.info("Successfully filled the missing sequence date.")

                except CustomException as e:
                        raise CustomException(e,sys)


                try:
                    #### Resampled by monthly
                    month_resampled_data = []
                    for unique_id in filled_missing_df['uniqueID'].unique():
                        mon_subset = filled_missing_df[filled_missing_df['uniqueID'] == unique_id]
                        monthly_resampled_ = mon_subset.groupby(['uniqueID', pd.Grouper(key='Booking_Date', freq='ME')]).sum().reset_index()
                        month_resampled_data.append(monthly_resampled_)
                    monthly_resampling_df = pd.concat(month_resampled_data) # Concatenate the filled subsets
                    logging.info("Successfully daily dataset has been resampled into monthly.")
                except CustomException as e:
                        raise CustomException(e,sys)


                ##### Resampled by weekly
                try:
                    weekly_filled_data = []
                    for unique_id_ in filled_missing_df['uniqueID'].unique():
                        week_subset = filled_missing_df[filled_missing_df['uniqueID'] == unique_id_]  ### Get each uniqueIDs in loop
                        filled_week_subset = self.weekly_resampling(week_subset)
                        weekly_filled_data.append(filled_week_subset)
                    
                    weekly_resampling_df = pd.concat(weekly_filled_data) # Concatenate the filled subsets
                    logging.info("Successfully resampled each unique ids in weekly.")
                except CustomException as e:
                        raise CustomException(e,sys)

                return filled_missing_df, weekly_resampling_df, monthly_resampling_df  

      
            selected_df_daily,selected_df_weekly,selected_df_monthly = sub_pre(selected_filtered_df)
            logging.info("Successfully saved selected daily, week and monthly resampled dataset into selected_df_daily,selected_df_weekly,selected_df_monthly variabels.")

            return selected_df_daily,selected_df_weekly,selected_df_monthly, max_date

        except CustomException as e:
            raise CustomException(e,sys)

    
    
"""Data preparation
Author: Lauro Reyes
"""
import logging
import datetime as dt
import json
import pandas as pd
from src.utils import (
    get_colum_by_type,
    read_file,
    has_rows,
    read_configuration
)
# log configuration
log_file_name = dt.datetime.strftime(dt.datetime.today(),'%Y%m%d_%H%M%S')
logging.basicConfig(filename=f'logs/prep_{log_file_name}.log', level=logging.DEBUG, filemode='w',
                    format='%(asctime)s:%(levelname)s:%(message)s')

def prep(config):
    """function to prepare the data"""
    # get data
    train_data = read_file(config['etl']['train_data'])
    test_data = read_file(config['etl']['test_data'])
    # check rows
    for table in [train_data,test_data]:
        has_rows(table)
    # drop 'SalePrice'
    x_train = pd.concat([train_data.drop(columns=['SalePrice']),test_data],ignore_index=True)
    #calculate the percentage of null values in the columns
    null_percent = x_train.isnull().sum()/x_train.shape[0]*100
    # deleting the columns with more than 50 missing values
    col_to_drop = null_percent[null_percent > 50].keys()
    x_train = x_train.drop(columns=list(col_to_drop))
    # feature engineering
    numerical_cols = get_colum_by_type(x_train)
    categorical_cols = get_colum_by_type(x_train,False)
    for column in numerical_cols:
        # Replace missing values with the mean
        x_train[column] = x_train[column].fillna(x_train[column].mean())
    for column in categorical_cols:
        # Replace missing values with the mode
        x_train[column] = x_train[column].fillna(x_train[column].mode()[0])
    if x_train.isnull().values.any():
        logging.warning("Train data has missing values")
    # One-hot encoding
    x_train = pd.get_dummies(data=x_train)
    # save the columns order
    with open(config['etl']['columns_file_path'], 'w', encoding='utf-8') as file:
        json.dump(x_train.columns.tolist(), file)
        logging.info("Columns order and lenght saved")
    # ex_trainport
    cut = train_data.shape[0]
    test_data_transform = x_train.iloc[cut:].copy()
    x_train_final = x_train.iloc[:cut].copy()
    test_data_transform.to_parquet(config['etl']['test_data_prep'])
    x_train_final.to_parquet(config['etl']['train_data_prep'])
    logging.info("Processed train and test data saved")

if __name__ == '__main__':
    project_config = read_configuration()
    prep(project_config)

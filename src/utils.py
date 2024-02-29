"""Suppor functions and libraries
Author: Lauro Reyes"""
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import yaml
import argparse
import logging
import sys
# get num and categorical columns
def get_colum_by_type(x_data,include=True):
    """Function to get the numerical or categorical columns from a dataset
        x_data: X data set
        include: True = return number types columns, False return categorical columns [default = True]
    """
    if include:
        result = (
            x_data.loc[:, x_data.isnull().any()]
            .select_dtypes(include='number')
            .columns
        )
        print("# Numerical columns with null values:", len(result))
        return result
    else:
        result = (
            x_data.loc[:, x_data.isnull().any()]
            .select_dtypes(exclude='number')
            .columns
        )
        print("# Categorical columns with null values:", len(result))
        return result

def has_rows(df,message="Table has no rows."):
    if len(df) == 0:
        logging.error(message)
        sys.exit(1)

def read_file(file,file_type='csv'):
    try:
        if file_type == 'csv':
            df = pd.read_csv(file)
            logging.info(f"{file} found and read successfully.")
            return df
        elif file_type == 'parquet':
            df = pd.read_parquet(file)
            logging.info(f"{file} found and read successfully.")
            return df
        elif file_type == 'joblib':
            joblib_object = joblib.load(file)
            logging.info(f"{file} found and read successfully.")
            return joblib_object
        else:
            sys.exit("Unknown file format")
    except FileNotFoundError:
        logging.exception(f"File not found")
        sys.exit(1)
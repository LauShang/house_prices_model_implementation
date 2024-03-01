"""Suppor functions and libraries
Author: Lauro Reyes"""
import logging
import json
import sys
import pandas as pd
import joblib
import yaml
# get num and categorical columns
def get_colum_by_type(x_data,include=True):
    """
    Function to get the numerical or categorical columns from a dataset.
    
    Parameters:
    - x_data: Dataset to process.
    - include (bool): If True, return numerical type columns. If False, return categorical columns.
                      Default is True.
    """
    if include:
        result = (
            x_data.loc[:, x_data.isnull().any()]
            .select_dtypes(include='number')
            .columns
        )
        print("# Numerical columns with null values:", len(result))
        return result
    result = (
        x_data.loc[:, x_data.isnull().any()]
        .select_dtypes(exclude='number')
        .columns
    )
    print("# Categorical columns with null values:", len(result))
    return result

def has_rows(df,message="Table has no rows."):
    """
    Checks if the given DataFrame has any rows.

    This function will log an error with a specified message and terminate
    the program if the DataFrame is empty.

    Parameters:
    - df (DataFrame): The DataFrame to check.
    - message (str, optional): The error message to log if the DataFrame is empty. 
                               Defaults to "Table has no rows.".
    """
    if len(df) == 0:
        logging.error(message)
        sys.exit()

def read_file(file,file_type='csv'):
    """
    Reads a file into a DataFrame or object based on the specified file type.

    This function attempts to read a file based on the given file_type parameter.
    It supports CSV, Parquet, and Joblib file formats. If the file is successfully
    read, it logs a message indicating success. For unsupported file formats,
    the function will terminate the program.

    Parameters:
    - file (str): The path to the file to read.
    - file_type (str, optional): The type of file to read ('csv', 'parquet', 'joblib'). 
        Defaults to 'csv'.

    Returns:
    - DataFrame or object: The content of the file read into a DataFrame 
        (for CSV and Parquet files) or any object stored in a Joblib file.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - SystemExit: If an unknown file format is provided.
    """
    try:
        if file_type == 'csv':
            df = pd.read_csv(file)
            logging.info("%s found and read successfully.",file)
            return df
        if file_type == 'parquet':
            df = pd.read_parquet(file)
            logging.info("%s found and read successfully.",file)
            return df
        if file_type == 'joblib':
            joblib_object = joblib.load(file)
            logging.info("%s found and read successfully.",file)
            return joblib_object
        sys.exit("Unknown file format")
    except FileNotFoundError:
        logging.exception("File not found")
        sys.exit()

def read_configuration(config_file="config.yaml"):
    """
    Read and load configuration from a YAML file.

    Args:
        config_file (str, optional): Path to the YAML configuration file. Defaults to "config.yaml".

    Returns:
        dict: Loaded configuration as a dictionary.

    Raises:
        FileNotFoundError: If the specified configuration file is not found.
        yaml.YAMLError: If there is an error in parsing the YAML file.
    """
    try:
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            logging.info("Yaml config found and read successfully.")
            return config
    except FileNotFoundError as e:
        logging.error("Failed to load configuration file: %s",e)
        sys.exit()
    except yaml.scanner.ScannerError as e:
        logging.error("Yaml config file has errors: %s",e)
        sys.exit()

def check_columns(df, json_file_path, df_name):
    """
    Check if the DataFrame columns match the order and length of the columns 
    saved in the columns JSON file.
    
    Args:
    - df: The pandas DataFrame to check.
    - json_file_path: Path to the JSON file containing the correct column order.
    
    Returns:
    - True if the DataFrame columns match the saved order, False otherwise.
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        saved_columns = json.load(file)
    # Check if the length and the order of the columns match
    if len(df.columns) == len(saved_columns) and all(df.columns == saved_columns):
        logging.info("Correct length and column order for %s",df_name)
    else:
        logging.error("Incorrect length or column order for %s",df_name)
        sys.exit()

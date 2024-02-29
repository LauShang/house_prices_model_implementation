"""Data preparation
Author: Lauro Reyes
"""
from src.utils import (
    pd,
    yaml,
    get_colum_by_type,
    logging
)
# log configuration
logging.basicConfig(filename='logs/prep.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def prep(config):
    """function to prepare the data"""
    # get data
    train_data = pd.read_csv(config['etl']['train_data'])
    test_data = pd.read_csv(config['etl']['test_data'])
    logging.info(f"Train and test data loaded. Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
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
        logging.warning(f"Train data has missing values")
    # One-hot encoding
    x_train = pd.get_dummies(data=x_train)
    # ex_trainport
    cut = train_data.shape[0]
    test_data_transform = x_train.iloc[cut:].copy()
    x_train_final = x_train.iloc[:cut].copy()
    test_data_transform.to_parquet(config['etl']['test_data_prep'])
    x_train_final.to_parquet(config['etl']['train_data_prep'])
    logging.info("Processed train and test data saved")

if __name__ == '__main__':
    try:
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        raise
    prep(config)

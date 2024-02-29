"""Data preparation
Author: Lauro Reyes
"""
from src.utils import (
    pd,
    yaml,
    argparse,
    get_colum_by_type
)

def prep(config):
    """function to prepare the data"""
    # get data
    train_data = pd.read_csv(config['etl']['train_data'])
    test_data = pd.read_csv(config['etl']['test_data'])
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
    if not x_train.isnull().values.any():
        print("\nThere are no missing values.")
    # One-hot encoding
    x_train = pd.get_dummies(data=x_train)
    # ex_trainport
    cut = train_data.shape[0]
    test_data_transform = x_train.iloc[cut:].copy()
    x_train_final = x_train.iloc[:cut].copy()
    test_data_transform.to_parquet(config['etl']['test_data_prep'])
    x_train_final.to_parquet(config['etl']['train_data_prep'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("-m","--manual", type=bool, default=False, help="Whether to manually set parameters.")
    parser.add_argument("--train_data", type=str, help="file name for train_data")
    parser.add_argument("--test_data", type=str, help="file name for test_data")
    parser.add_argument("--test_data_prep", type=str, help="file name for test_data_prep")
    parser.add_argument("--train_data_prep", type=str, help="file name for train_data_prep")
    # Parse arguments
    args = parser.parse_args()
    # Conditionally load config or set parameters based on manual input
    if args.manual:
        config = {}
        values = ['train_data', 'test_data', 'test_data_prep', 'train_data_prep']
        objects = [args.train_data, args.test_data, args.test_data_prep, args.train_data_prep]
        # etl parameters
        config['etl'] = {}
        for value,object_ in zip(values,objects):
            if object_== None:
                raise ValueError(f"{value} must be assigned")
            config['etl'][value] = object_
    else:
        # Open YAML config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
    prep(config)

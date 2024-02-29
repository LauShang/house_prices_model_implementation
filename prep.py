"""Data preparation
Author: Lauro Reyes
"""
import pandas as pd

def prep():
    """Main procces"""
    # get data
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    # drop 'SalePrice'
    x_train = pd.concat([train_data.drop(columns=['SalePrice']),test_data],ignore_index=True)
    #calculate the percentage of null values in the columns
    null_percent = x_train.isnull().sum()/x_train.shape[0]*100
    # deleting the columns with more than 50 missing values
    col_to_drop = null_percent[null_percent > 50].keys()
    x_train = x_train.drop(columns=list(col_to_drop))
    # feature engineering
    numerical_cols = (
        x_train.loc[:, x_train.isnull().any()]
        .select_dtypes(include='number')
        .columns
    )
    categorical_cols = (
        x_train.loc[:, x_train.isnull().any()]
        .select_dtypes(exclude='number')
        .columns
    )
    print("# Numerical columns with null values:", len(numerical_cols))
    print("# Categorical columns with null values:", len(categorical_cols))
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
    test_data_transform.to_parquet('./data/test_prep.parquet')
    x_train_final.to_parquet('./data/train_prep.parquet')

if __name__ == '__main__':
    prep()

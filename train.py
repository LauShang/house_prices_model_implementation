"""Script for train a randomforest model"""
import argparse
import logging
import datetime as dt
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from src.utils import (
    has_rows,
    read_file,
    read_configuration,
    check_columns
)
# log configuration
log_file_name = dt.datetime.strftime(dt.datetime.today(),'%Y%m%d_%H%M%S')
logging.basicConfig(filename=f'logs/train_{log_file_name}.log', level=logging.DEBUG, filemode='w',
                    format='%(asctime)s:%(levelname)s:%(message)s')

def train(config):
    """Function to train the RF Model"""
    # import data
    x_prep = read_file(config['etl']['train_data_prep'],'parquet')
    train_data = read_file(config['etl']['train_data'])
    # check if data has rows
    has_rows(x_prep,"The train_data_prep file has no rows")
    has_rows(train_data,"The train_data file has no rows")
    # check column order
    check_columns(x_prep,config['etl']['columns_file_path'],'x_prep')
    # adjust columns
    x_prep = x_prep.drop(columns=['Id'])
    y_obj = train_data['SalePrice']
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x_prep,
        y_obj,
        test_size=config['modeling']['test_size'],
        random_state=config['modeling']['random_seed']
    )
    # Standardize the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    # Build the Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=config['modeling']['random_forest']['n_estimators'],
        random_state=config['modeling']['random_seed'],
        max_depth=config['modeling']['random_forest']['max_depth']
        )
    # Train the model
    rf_model.fit(x_train_scaled, y_train)
    logging.info('Random Forest Model fitted')
    # Save the scaler
    joblib.dump(scaler, config['modeling']['scaler_file'])
    # Save the model
    joblib.dump(rf_model, config['modeling']['model_file'])
    y_test_pred_rf = rf_model.predict(x_test_scaled)
    rmse_rf = np.sqrt(mean_squared_error(np.log(y_test),np.log(y_test_pred_rf)))
    rounded_rmse_rf = round(rmse_rf, 4)
    logging.info('Root Mean Squared Error on validation Set (Random Forest): %d',rounded_rmse_rf)

if __name__ == '__main__':
    # Open YAML config file
    global_config = read_configuration()
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("-m","--mod_model", type=bool, default=False,
                        help="Whether to manually set parameters.")
    parser.add_argument("--n_estimators", type=int,
                        help="The number of trees in the forest. "
                        "A higher number increases model complexity "
                        "and potential for overfitting.")
    parser.add_argument("--random_seed", type=int,
                        help="Seed for the random number generator. "
                        "Ensures reproducibility of model results.")
    parser.add_argument("--max_depth", type=int,
                        help="The maximum depth of the trees. Limits the complexity of the model"
                        " to prevent overfitting. Use 'None' for unlimited depth.")
    # Parse arguments
    args = parser.parse_args()
    # Conditionally load config or set parameters based on manual input
    if args.mod_model:
        values = ['n_estimators', 'random_seed', 'max_depth']
        objects = [args.n_estimators, args.random_seed, args.max_depth]
        # model parameters
        for value,object_ in zip(values,objects):
            if object_ is None:
                logging.error("The custom parameter %s was not assigned",value)
                raise ValueError(f"{value} must be assigned")
            if value == 'random_seed':
                global_config['modeling'][value] = object_
            else:
                global_config['modeling']['random_forest'][value] = object_
    train(global_config)

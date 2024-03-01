"""Script for make a prediction with a randomforest model"""
import logging
import datetime as dt
import pandas as pd
from src.utils import (
    read_file,
    read_configuration,
    check_columns
)
# log configuration
log_file_name = dt.datetime.strftime(dt.datetime.today(),'%Y%m%d_%H%M%S')
logging.basicConfig(filename=f'logs/inference_{log_file_name}.log', level=logging.DEBUG,
                    filemode='w', format='%(asctime)s:%(levelname)s:%(message)s')

def inference(config):
    "return values for test data"
    # Load the scaler
    scaler = read_file(config['modeling']['scaler_file'],'joblib')
    # Load the model
    rf_model = read_file(config['modeling']['model_file'],'joblib')
    # sale transformed test data
    test_data_transform = read_file(config['etl']['test_data_prep'],'parquet')
    # check columns
    check_columns(test_data_transform,config['etl']['columns_file_path'],'test_data_transform')
    test_x = test_data_transform.drop('Id', axis=1)
    test_x_scaled = scaler.transform(test_x)
    test_preds_rf = rf_model.predict(test_x_scaled)
    # Save the predictions to a CSV file
    test_data = read_file(config['etl']['test_data'])
    result = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_preds_rf})
    result.to_csv(config['etl']['predictions'], index=False)
    logging.info('Model predictions updated')


if __name__ == '__main__':
    # open yaml
    global_config = read_configuration()
    inference(global_config)

"""Script for make a prediction with a randomforest model"""
from src.utils import (
    pd,
    yaml,
    read_file,
    logging
)
# log configuration
logging.basicConfig(filename='logs/inference.log', level=logging.DEBUG, filemode='w',
                    format='%(asctime)s:%(levelname)s:%(message)s')

def inference(config):
    "return values for test data"
    # Load the scaler
    scaler = read_file(config['modeling']['scaler_file'],'joblib')
    # Load the model
    rf_model = read_file(config['modeling']['model_file'],'joblib')
    # sale transformed test data
    test_data_transform = read_file(config['etl']['test_data_prep'],'parquet')
    test_x = test_data_transform.drop('Id', axis=1)
    test_x_scaled = scaler.transform(test_x)
    test_preds_rf = rf_model.predict(test_x_scaled)
    # Save the predictions to a CSV file
    test_data = read_file(config['etl']['test_data'])
    result = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_preds_rf})
    result.to_csv(config['etl']['predictions'], index=False)
    logging.info(f'Model predictions updated')


if __name__ == '__main__':
    # open yaml
    with open("config.yaml", "r") as file:
        global_config = yaml.safe_load(file)
    inference(global_config)

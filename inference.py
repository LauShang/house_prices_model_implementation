"""Script for make a prediction with a randomforest model"""
from src.utils import (
    joblib,
    pd,
    yaml
)

def inference():
    "return values for test data"
    # open yaml
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    # Load the scaler
    scaler = joblib.load(config['modeling']['scaler_file'])
    # Load the model
    rf_model = joblib.load(config['modeling']['model_file'])
    # sale transformed test data
    test_data_transform = pd.read_parquet(config['etl']['test_data_prep'])
    test_x = test_data_transform.drop('Id', axis=1)
    test_x_scaled = scaler.transform(test_x)
    test_preds_rf = rf_model.predict(test_x_scaled)
    # Save the predictions to a CSV file
    test_data = pd.read_csv(config['etl']['test_data'])
    result = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_preds_rf})
    result.to_csv(config['etl']['predictions'], index=False)

if __name__ == '__main__':
    inference()

"""Script for make a prediction with a randomforest model"""
import pandas as pd
import joblib

def inference():
    "main process"
    # Load the scaler
    scaler = joblib.load('data/scaler.joblib')
    # Load the model
    rf_model = joblib.load('data/model.joblib')
    # sale transformed test data
    test_data_transform = pd.read_parquet('./data/test_prep.parquet')
    test_x = test_data_transform.drop('Id', axis=1)
    test_x_scaled = scaler.transform(test_x)
    test_preds_rf = rf_model.predict(test_x_scaled)
    # Save the predictions to a CSV file
    test_data = pd.read_csv('./data/test.csv')
    result = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_preds_rf})
    result.to_csv('data/predictions.csv', index=False)


if __name__ == '__main__':
    inference()

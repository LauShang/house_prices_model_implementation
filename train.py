"""Script for train a randomforest model"""
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def train():
    """Main process"""
    # import data
    x_prep = pd.read_parquet('./data/train_prep.parquet')
    train_data = pd.read_csv('./data/train.csv')
    # adjust columns
    x_prep = x_prep.drop(columns=['Id'])
    y_obj = train_data['SalePrice']
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x_prep,
        y_obj,
        test_size=0.2,
        random_state=42
    )
    # Standardize the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    # Build the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    # Train the model
    rf_model.fit(x_train_scaled, y_train)
    # Save the scaler
    joblib.dump(scaler, 'data/scaler.joblib')
    # Save the model
    joblib.dump(rf_model, 'data/model.joblib')
    y_test_pred_rf = rf_model.predict(x_test_scaled)
    rmse_rf = np.sqrt(mean_squared_error(np.log(y_test),np.log(y_test_pred_rf)))
    rounded_rmse_rf = round(rmse_rf, 4)
    print(f'Root Mean Squared Error on test Set (Random Forest): {rounded_rmse_rf}')

if __name__ == '__main__':
    train()

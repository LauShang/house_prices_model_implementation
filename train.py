"""Script for train a randomforest model"""
from src.utils import (
    joblib,
    pd,
    np,
    mean_squared_error,
    train_test_split,
    StandardScaler,
    RandomForestRegressor,
    yaml
)

def train():
    """Function to train the RF Model"""
    # open yaml
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    # import data
    x_prep = pd.read_parquet(config['etl']['train_data_prep'])
    train_data = pd.read_csv(config['etl']['train_data'])
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
    # Save the scaler
    joblib.dump(scaler, config['modeling']['scaler_file'])
    # Save the model
    joblib.dump(rf_model, config['modeling']['model_file'])
    y_test_pred_rf = rf_model.predict(x_test_scaled)
    rmse_rf = np.sqrt(mean_squared_error(np.log(y_test),np.log(y_test_pred_rf)))
    rounded_rmse_rf = round(rmse_rf, 4)
    print(f'Root Mean Squared Error on test Set (Random Forest): {rounded_rmse_rf}')

if __name__ == '__main__':
    train()

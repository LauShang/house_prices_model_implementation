"""Script for train a randomforest model"""
from src.utils import (
    joblib,
    pd,
    np,
    mean_squared_error,
    train_test_split,
    StandardScaler,
    RandomForestRegressor,
    yaml,
    argparse,
    logging,
    has_rows,
    sys
)
# log configuration
logging.basicConfig(filename='logs/train.log', level=logging.DEBUG, filemode='w',
                    format='%(asctime)s:%(levelname)s:%(message)s')

def train(config):
    """Function to train the RF Model"""
    # import data
    x_prep = pd.read_parquet(config['etl']['train_data_prep'])
    train_data = pd.read_csv(config['etl']['train_data'])
    # check if data has rows
    has_rows(x_prep,"The train_data_prep file has no rows")
    has_rows(train_data,"The train_data file has no rows")
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
    logging.info(f'Root Mean Squared Error on test Set (Random Forest): {rounded_rmse_rf}')

if __name__ == '__main__':
    # Open YAML config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("-m","--mod_model", type=bool, default=False, help="Whether to manually set parameters.")
    parser.add_argument("--n_estimators", type=int, help="The number of trees in the forest. A higher number increases model complexity and potential for overfitting.")
    parser.add_argument("--random_seed", type=int, help="Seed for the random number generator. Ensures reproducibility of model results.")
    parser.add_argument("--max_depth", type=int, help="The maximum depth of the trees. Limits the complexity of the model to prevent overfitting. Use 'None' for unlimited depth.")
    # Parse arguments
    args = parser.parse_args()
    # Conditionally load config or set parameters based on manual input
    if args.mod_model:
        values = ['n_estimators', 'random_seed', 'max_depth']
        objects = [args.n_estimators, args.random_seed, args.max_depth]
        # model parameters
        for value,object_ in zip(values,objects):
            if object_== None:
                logging.error(f"The custom parameter {value} was not assigned")
                raise ValueError(f"{value} must be assigned")
            else:
                if value == 'random_seed':
                    config['modeling'][value] = object_
                else:
                    config['modeling']['random_forest'][value] = object_
    train(config)

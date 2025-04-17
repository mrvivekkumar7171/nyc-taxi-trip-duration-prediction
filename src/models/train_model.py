# train_model.py
import pathlib
import sys
import joblib
import mlflow
import pandas as pd
from hyperopt import hp
from sklearn.model_selection import train_test_split
from hyperopt.pyll.base import scope
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from xgboost import XGBRegressor

# Performing hyperparameter tuning (grid search and random search if used must be there)

def find_best_model_with_params(X_train, y_train, X_test, y_test):
    # In this setup we are actually manually finding the best model and if the model don't exist in the below
    # hyperparameters dictionary then we add it (we don't delete it)
    hyperparameters = {
        "RandomForestRegressor": {
            "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
            "max_depth": hp.choice("max_depth", [6, 8, 10]),
            "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
        },
        "XGBRegressor": {
            "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
            "max_depth": hp.choice("max_depth", [6, 8, 10]),
            "learning_rate": hp.uniform("learning_rate", 0.03, 0.3),
        },###### add new model entry here and space of it
    }

    # evaluate the loss of the model to find the best hyperparameters
    def evaluate_model(hyperopt_params):
        params = hyperopt_params
        # fail safe check for the model's hyperopt parameters
        if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
        if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) 
        if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])

        ###### we are required to change the name of model here
        # Loading the model
        model = XGBRegressor(**params) # here the XGBRegressor is hard coded and need to change
        # training the model
        model.fit(X_train, y_train)
        # making prediction
        y_pred = model.predict(X_test)

        # calculating the RMSE as our example is regression example
        model_rmse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric('RMSE', model_rmse)  # record actual metric with mlflow run
        loss = model_rmse  
        return {'loss': loss, 'status': STATUS_OK}

    ###### we are required to change the name of model here also
    # providing the relevant space
    space = hyperparameters['XGBRegressor']
    # logging the model in mlflow after starting a mlflow run to find the best parameters
    with mlflow.start_run(run_name='XGBRegressor'): ###### we are required to change the name of model here also
        argmin = fmin( # fmin to reduce the loss
            fn=evaluate_model, # the function to reduce
            space=space, # space of the model
            algo=tpe.suggest, # algorithm to use for the search
            # we can also set the max_evals into the params.yaml and read it from there as a parameter
            max_evals=5, # number of iterations to run (in actual project it can go from 50 to 500)
            trials=Trials(), # 
            verbose=True # to debug the model
            )
    run_ids = []
    ###### we are required to change the name of model here also
    # running the mlflow with the best hyperparameter we have got from the above experiment of mlflow run
    with mlflow.start_run(run_name='XGB Final Model') as run:
        # Programmatically accessing the best parameters and run
        run_id = run.info.run_id
        run_name = run.data.tags['mlflow.runName']
        run_ids += [(run_name, run_id)]
        
        # configure params (again the same steps as above)
        params = space_eval(space, argmin)
        if 'max_depth' in params: params['max_depth']=int(params['max_depth'])       
        if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight'])
        if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])  
        mlflow.log_params(params)

        # Train your machine learning model
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        # saving the best model in the mlflow
        mlflow.sklearn.log_model(model, 'model')  # persist model with mlflow for registering
    return model


def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + "/model.joblib")


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1] # data/processed/
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + "/models" # save the output model in models folder
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) # if the folder don't exist then it will create it

    # creating model on train dataset
    TARGET = "trip_duration"

    train_features = pd.read_csv(data_path + "/train.csv")
    X = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    trained_model = find_best_model_with_params(X_train, y_train, X_test, y_test)
    save_model(trained_model, output_path)
    # We will push this model to S3 and also copy in the root folder for Dockerfile to pick


if __name__ == "__main__":
    main()
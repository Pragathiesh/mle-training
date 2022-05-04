"""
This module contains helper functions to score the models.
Can be run standalone with commandline arguments for models and the datasets to load and show them.
"""
import argparse
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

remote_server_uri = "http://0.0.0.0:5000"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
exp_name = "Housing_mle-training"
mlflow.set_experiment(exp_name)


def get_path():
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle-training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        help="path to the datasets ",
        type=str,
        default="data/processed"
    )
    parser.add_argument(
        "--modelpath",
        help="path to the model files ",
        type=str,
        default="artifacts"
    )
    return parser.parse_args()


exp_name = "Housing_mle-training"
mlflow.set_experiment(exp_name)


def scoring(strat_test_set, lin_reg, tree_reg, forest_reg, grid_search):
    """Computes the Scores of the given model on given data.
    """
    # housing_prepared,housing_labels,strat_test_set=preprocess(housing)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    X_test_prepared = imputer.fit_transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(
            X_test_cat, drop_first=True))

    lin_predictions = lin_reg.predict(X_test_prepared)
    lin_mse = mean_squared_error(y_test, lin_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(y_test, lin_predictions)

    tree_predictions = tree_reg.predict(X_test_prepared)
    tree_mse = mean_squared_error(y_test, tree_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = mean_absolute_error(y_test, tree_predictions)

    forest_predictions = forest_reg.predict(X_test_prepared)
    forest_mse = mean_squared_error(y_test, forest_predictions)
    forest_rmse = np.sqrt(forest_mse)
    forest_mae = mean_absolute_error(y_test, forest_predictions)

    grid_search_predictions = grid_search.predict(X_test_prepared)
    grid_search_mse = mean_squared_error(y_test, grid_search_predictions)
    grid_search_rmse = np.sqrt(grid_search_mse)
    grid_search_mae = mean_absolute_error(y_test, grid_search_predictions)

    lin_scores = [lin_mae, lin_mse, lin_rmse]
    tree_scores = [tree_mae, tree_mse, tree_rmse]
    forest_scores = [forest_mae, forest_mse, forest_rmse]
    grid_search_scores = [grid_search_mae, grid_search_mse, grid_search_rmse]

    return lin_scores, tree_scores, forest_scores, grid_search_scores


def rem_index(data):
    new_columns = data.columns.values
    new_columns[0] = ""
    data.columns = new_columns
    data = data.set_index("")
    return data


def load_models(model_path):
    """Loads models from given directory path.
    """
    model_names = [
        "lin_model",
        "tree_model",
        "forest_model",
        "grid_search_model"]
    models = []
    for i in model_names:
        with open(model_path + "/" + i + "/model.pkl", "rb") as f:
            models.append(pickle.load(f))
    return models


def mlflow_score(models):
    """Scores given model on given data.
    """

    with mlflow.start_run(run_name="SCORE"):
        testset = pd.read_csv(data_path + "/train_set.csv")
        testset = rem_index(testset)

        lin_scores, tree_scores, forest_scores, grid_search_scores = scoring(
            testset, models[0], models[1], models[2], models[3]
        )
        print(lin_scores, tree_scores, forest_scores, grid_search_scores)
        mlflow.log_metrics(
            {
                "lin_mae": lin_scores[0],
                "tree_mae": tree_scores[0],
                "forest_mae": forest_scores[0],
                "grid_search_mae": grid_search_scores[0],
            }
        )
        mlflow.log_metrics(
            {
                "lin_mse": lin_scores[1],
                "tree_mse": tree_scores[1],
                "forest_mse": forest_scores[1],
                "grid_search_mse": grid_search_scores[1],
            }
        )
        mlflow.log_metrics(
            {
                "lin_rmse": lin_scores[2],
                "tree_rmse": tree_scores[2],
                "forest_rmse": forest_scores[2],
                "grid_search_rmse": grid_search_scores[2],
            }
        )


if __name__ == "__main__":
    """Runs the whole scoring process according to the given commandline arguments.
    """
    args = parse_args()
    path_parent = get_path()
    data_path = path_parent + "/" + args.datapath
    model_path = path_parent + "/" + args.modelpath
    models = load_models(model_path)
    mlflow_score(models)

"""
This module contains helper functions to train models.
Can be run standalone with commandline arguments for dataset path and models directory.
"""
import argparse
import os
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

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
    """Commandline argument parser for standalone run.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputpath",
        help="path to the input dataset ",
        type=str,
        default="data/processed/",
    )
    parser.add_argument(
        "--outputpath", help="path to store the output ", type=str, default="artifacts"
    )
    return parser.parse_args()


def train(housing_prepared, housing_labels):
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    forest_reg.fit(housing_prepared, housing_labels)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    return lin_reg, tree_reg, forest_reg, grid_search


def rem_index(data):
    new_columns = data.columns.values
    new_columns[0] = ""
    data.columns = new_columns
    data = data.set_index("")
    return data


def load_data(in_path):
    """Loads dataset and splits features and labels.
    """
    prepared = pd.read_csv(in_path + "/train_X.csv", index_col=False)
    prepared = rem_index(prepared)
    lables = pd.read_csv(in_path + "/train_y.csv", index_col=False)
    lables = lables.values.ravel()
    return prepared, lables


def rem_artifacts(out_path):
    model_names = ["lin_model", "tree_model", "forest_model", "grid_search_model"]
    for i in model_names:
        if os.path.exists(out_path + "/" + i):
            shutil.rmtree(out_path + "/" + i)


def mlflow_model(lin_reg, tree_reg, forest_reg, grid_search, out_path):
    """Saves the given model in artifacts directory as pickle file.
    """
    with mlflow.start_run(run_name="TRAIN"):
        mlflow.sklearn.save_model(lin_reg, out_path + "/lin_model")
        mlflow.sklearn.save_model(tree_reg, out_path + "/tree_model")
        mlflow.sklearn.save_model(forest_reg, out_path + "/forest_model")
        mlflow.sklearn.save_model(grid_search, out_path + "/grid_search_model")


if __name__ == "__main__":
    """Runs the whole training process according to given commandline arguments.
    """
    args = parse_args()
    path_parent = get_path()
    in_path = path_parent + args.inputpath
    out_path = path_parent + args.outputpath
    rem_artifacts(out_path)
    prepared, labels = load_data(in_path)
    lin_reg, tree_reg, forest_reg, grid_search = train(prepared, labels)
    mlflow_model(lin_reg, tree_reg, forest_reg, grid_search, out_path)

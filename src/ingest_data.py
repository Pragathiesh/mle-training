"""
This module contains helper functions for ingestion of data.
Running this standalone downloads the housing data and stores preprocessed copies of it in the specified folders.
"""
import argparse
import os
import tarfile
from importlib.resources import path

import mlflow
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data/raw", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
imputer = SimpleImputer(strategy="median")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Function to download and extract housing data.

    Parameters
    ----------
    housing_url : str
        Url to download the housing data from.
    housing_path : str
        Path to store the raw csv files after extraction.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def preprocess(housing):

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    """Does stratified shuffle split on "income_cat" attribute of housing data."""

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]

    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    """Preprocesses the given dataframe. Imputes missing values with median.
    Replaces categorical column "ocean_proximity" with onehot dummy variables.
    """

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    return housing_prepared, housing_labels, strat_test_set


def parse_args():

    """Commandline argument parser for standalone run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        help="path to store the dataset ",
        type=str,
        default="data/raw/housing",
    )
    return parser.parse_args()


def get_path():

    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle-training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


data_pro = "data/processed/"
if __name__ == "__main__":
    """Does all the ingesting work (fetching, splitting, preprocessing).
    Gets called if this module is run standalone.
    """
    args = parse_args()

    path_parent = get_path()
    path = path_parent + args.datapath
    fetch_housing_data(housing_path=path)

    data = load_housing_data(housing_path=path)

    prep, lab, testset = preprocess(data)

    prep.to_csv(path_parent + "/" + data_pro + "train_X.csv")
    lab.to_csv(path_parent + "/" + data_pro + "train_y.csv", index=False)
    testset.to_csv(path_parent + "/" + data_pro + "train_set.csv")

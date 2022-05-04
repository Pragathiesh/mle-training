"""
This module contains unit test for src/housing_price/ingest_data.py
"""
import os
from importlib.resources import path

import pandas as pd
from src import ingest_data as data

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data/raw", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

args = data.parse_args()
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = args.datapath
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def test_parse_args():
    """
    Tests parse_args function.
    """
    assert args.datapath == "data/raw/housing"


rootpath = data.get_path()


def test_fetch_data():
    """
    Tests fetch_housing_data function.
    """
    data.fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    assert os.path.isfile(rootpath + args.datapath + "/housing.tgz")
    assert os.path.isfile(rootpath + args.datapath + "/housing.csv")


def test_preprocess():
    """
    Tests train_test_split function.
    """
    housing_df = pd.read_csv(rootpath + args.datapath + "/housing.csv")
    (
        train_X,
        train_y,
        test_set,
    ) = data.preprocess(housing_df)
    assert len(train_X) == len(housing_df) * 0.8
    assert len(test_set) == len(housing_df) * 0.2
    assert "income_cat" not in train_X.columns
    assert "income_cat" not in test_set.columns

    cats = housing_df["ocean_proximity"].unique()

    assert not train_X.isna().sum().sum()
    assert "ocean_proximity" not in train_X.columns
    assert "ocean_proximity" in test_set.columns
    assert "rooms_per_household" in train_X.columns
    assert "rooms_per_household" not in test_set.columns
    assert "population_per_household" in train_X.columns
    assert "population_per_household" not in test_set.columns
    assert "bedrooms_per_room" in train_X.columns
    assert "bedrooms_per_room" not in test_set.columns


test_parse_args()
test_preprocess()
test_preprocess()

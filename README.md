# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest
 - Best model using Grid Search

## Installation:
### Prerequisites:
Prerequisite dependencies are stored in `deploy/conda/linux_cpu_py39.yml`. To setup the conda environment:

`$ conda env create --file deploy/conda/linux_cpu_py39.yml`

`$ conda activate mle-dev`


## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script

### To download and process data:
`$ python src/ingest_data.py`
### To train the models:
`$ python src/train.py`
### To score trained models:
`$ python src/score.py`
### Note:
You can get information on command line arguments for each of the above scripts using `-h` or `--help`. For example: 

`$ python src/train.py --help`
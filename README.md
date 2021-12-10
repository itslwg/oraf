
# 'ORAF: Ordinal Regression for Avalanche Forecasting'

Authors:
- Ludvig Wärnberg Gerdin
- David Howard Neill
- Nathan Simonis

## Installation

```console
pip install -r requirements.txt
```
or 
```console
pip install [package name]
```
## Data

This folder contains the data as delivered to us, including 4 csv files from measuring stations at different locations in Switzerland.  Also included is the final csv file generated once all the processing has been completed.  Finally, there is a text file that describes each of the variables.

## Scipts
This folder contains the scripts that are used across the different notebooks.  These include the full data processing pipeline that is used to transform the raw data into the data ultimately used by the models and performance metrics implementations.

### Models
This folder consists of 3 files that implement the different methods we have used.  Its function is to take as input the cleaned data, go through the training procedure and then provide forecasts for the period from 2015 to 2020.

## Cross Validation
3 subfolders exist in this folder and represent the different notebooks per method that have been used to carry out the cross-validation procedure.  The parameters of the final models have been saved therein.

## Notebooks
This folder contains the main notebook that imports raw data, cleans it, trains each model based on the previously specified parameters and displays the final results in a table. By running this notebook, the figures and scores given in the report are reproduced.

The methods included in each script are reused in the notebooks and should consequently be kept in place. 

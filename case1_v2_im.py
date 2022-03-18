from sklearn.tree import DecisionTreeRegressor
from random import random
from tabnanny import verbose
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit
import sklearn.preprocessing as preproc
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from importlib.resources import path
import Helpers.ErrorMetrics
import Helpers.DataPreprocessing as hdp
import os
import sys
script_path = os.path.abspath('')
sys.path.append(script_path)
sys.path.append(os.path.join(script_path, 'Helpers'))

sns.set()


# Read data
file_name = 'Realized Schedule 20210101-20220228.xlsx'
raw_data = pd.read_excel(file_name)

# Transform AircraftType in only strings
raw_data.loc[:, ('AircraftType')] = raw_data.loc[:,
                                                 ('AircraftType')].astype(str)

raw_attributes_used = [
    'ScheduleTime',
    "Airline",
    'Destination',
    'FlightType',
    'Sector',
]


input_data = raw_data[raw_attributes_used]
transformer = hdp.DataTransformer(top=3, dummy_encode=True)
transformer.fit(input_data)

X_transformed_df = transformer.transform(input_data)

# Be careful for actual passengers = 0
X_df = X_transformed_df[raw_data.LoadFactor != 0]
load_factor = raw_data[raw_data.LoadFactor != 0].LoadFactor
seat_capacity = raw_data[raw_data.LoadFactor != 0].SeatCapacity
y_df = pd.DataFrame(data=load_factor.values *
                    seat_capacity.values, columns=['Number of Passengers'])

X = X_df.values
y = y_df.values.ravel()

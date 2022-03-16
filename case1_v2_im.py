import Helpers.DataPreprocessing as hdp
from importlib.resources import path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preproc
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import pandas as pd
import numpy as np
from tabnanny import verbose
from random import random
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
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

raw_attributes_used = ['ScheduleTime', "Airline", 'Destination',
                       'AircraftType', 'FlightType', 'Sector']

raw_categorical_attributes_used = [
    'Airline', 'Destination', 'AircraftType', 'FlightType', 'Sector']


num_pipeline = Pipeline([
    ('data_transformer', hdp.AtemporalEncodingFeaturesTransformer(
        raw_data, raw_attributes_used, raw_categorical_attributes_used, 10))
])

data = num_pipeline.fit_transform(raw_data)
X, y = data.values[:, :-1], data.values[:, -1]
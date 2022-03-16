import os
import sys
from random import random
from tabnanny import verbose
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import sklearn.preprocessing as preproc
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
import dateutil.easter

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_holidays(data_frame_in, easter_days_off=3, christmas_weeks_numbers=[52, 53, 1]):
    # Easter
    years = np.asarray(data_frame_in.ScheduleTime.dt.year)
    easter_index = [dateutil.easter.easter(year) for year in years]
    easter_index = np.array(np.abs(
        easter_index - data_frame_in.ScheduleTime.dt.date).dt.days <= easter_days_off)

    # Christmas - New Year's Eve
    weeks = np.asarray(data_frame_in.ScheduleTime.dt.isocalendar().week)
    christmas_index = np.array(
        [weeks[i] in christmas_weeks_numbers for i in range(data_frame_in.shape[0])])

    # merge the indixes
    indx = christmas_index | easter_index
    temp_df = {'Holiday': indx.astype(int)}
    temp_df = pd.DataFrame(temp_df)
    return indx.astype(int)


def prepare_data(df, attributes_used, top_cat_dictionary=None):
    df = df[attributes_used].copy()
    attrs = attributes_used.copy()

    df['Year'] = df['ScheduleTime'].dt.isocalendar().year
    attrs.append('Year')

    df['WeekNumber'] = df['ScheduleTime'].dt.isocalendar().week
    attrs.append('WeekNumber')

    df['Day'] = df['ScheduleTime'].dt.isocalendar().day
    attrs.append('Day')

    df['Hour'] = df['ScheduleTime'].dt.hour
    attrs.append('Hour')

    df['Holiday'] = get_holidays(df)
    attrs.append('Holiday')

    df.drop('ScheduleTime', axis=1, inplace=True)

    if top_cat_dictionary is None:
        return df

    for attr in top_cat_dictionary.keys():
        for i in top_cat_dictionary[attr]:
            feature_name = attr + '_' + str(i)
            df[feature_name] = (df[attr] == i).astype(int)
            attrs.append(feature_name)
        df.drop(attr, axis=1, inplace=True)

    return df


def get_categorical_encoding(df_in, categorical_attributes, top_values):
    cat_dict = {}
    for attr in categorical_attributes:
        top = top_values
        if len(df_in[attr].value_counts()) < top:
            top = len(df_in[attr].value_counts())
        cat_dict[attr] = df_in[attr].value_counts().head(top).index
    return cat_dict


# Read data
file_name = 'Realized Schedule 20210101-20220228.xlsx'
raw_data = pd.read_excel(file_name)

print('Null values:')
print(raw_data.isnull().sum())

print('Features:')
print(raw_data.columns)

# Transform AircraftType in only strings
raw_data.loc[:, ('AircraftType')] = raw_data.loc[:,
                                                 ('AircraftType')].astype(str)

# Select which attributes to use (left out bc not useful or target variable, LoadFactor)
# Seat capacity has corr only 0.066637


# raw_attributes_used = ['ScheduleTime', "Airline", 'Destination',
#                        'AircraftType', 'FlightType', 'Sector', 'SeatCapacity']

raw_attributes_used = ['ScheduleTime', "Airline", 'Destination',
                       'AircraftType', 'FlightType', 'Sector']

raw_categorical_attributes_used = [
    'Airline', 'Destination', 'AircraftType', 'FlightType', 'Sector']

top_used = 10
cat_dict = get_categorical_encoding(raw_data,
                                    categorical_attributes=raw_categorical_attributes_used, top_values=top_used)

# Remove any types other than regular J
cat_dict['FlightType'] = cat_dict['FlightType'][:-2]

# Create a (deep) copy of the data set to process
X = prepare_data(raw_data, raw_attributes_used,
                 cat_dict)
y = pd.DataFrame(raw_data['LoadFactor'])

data = pd.concat((X, y), axis=1)

####################################################################

####################################################################

data_train, data_test = train_test_split(data, test_size=0.2)

X_train, X_test = data_train.values[:, :-1], data_test.values[:, :-1]
y_train, y_test = data_train.values[:, -1], data_test.values[:, -1]


def rmse(actual, pred):
    res = (actual-pred)**2
    return np.sqrt(res.mean())


# regr = RandomForestRegressor()
# regr.fit(X_train, y_train)

# y_hat = regr.predict(X_test)
# rmse(y_test, y_hat)

# models = []

# grid_param = {
#     'alpha':[0.00046415888336127773],
#     'l1_ratio':[np.linspace(0.015, 0.15, 10)],
# }
# regr = GridSearchCV(ElasticNet(), grid_param, cv=5, n_jobs=-1, verbose=2)
# regr.fit(X_train, y_train)
# models.append(regr.best_estimator_)
# elastNet  = ElasticNet(alpha = 0.0004641588, l1_ratio=0.03)
# elastNet.fit(X_train, y_train)
# y_hat = elastNet.predict(X_test)
# error = rmse(y_hat, y_test)
K = 10  # CV Folds
CV = KFold(n_splits=K)

errors = np.zeros(K)
estimators = []
for i, (train_index, test_index) in enumerate(CV.split(X_train, y_train)):
    X_train_set = X_train[train_index]
    y_train_set = y_train[train_index]
    X_test_set = X_train[test_index]
    y_test_set = y_train[test_index]

    param_grid = {
        'l1_ratio': [0.01, 0.05, 0.1],
        'alpha': [0.0004, 0.0005, 0.001]
    }

    est = GridSearchCV(ElasticNet(), param_grid=param_grid, cv=10)
    est.fit(X_train_set, y_train_set)
    print(i)
    estimators.append(est.best_estimator_)
    errors[i] = rmse(y_test_set, est.best_estimator_.predict(X_test_set))

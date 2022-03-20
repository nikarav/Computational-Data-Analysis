from pydoc import Helper
import sched
from sklearn.tree import DecisionTreeRegressor
from random import random
from tabnanny import verbose
import numpy as np
import pandas as pd
from scipy import stats
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
import Helpers.SamplingPreprocessing as spreproc
import os
import sys
import folium
import imageio
from tqdm import tqdm_notebook
from folium.plugins import MarkerCluster
import imageio
import mapclassify as mc
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import scipy
from itertools import product
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['font.serif'] = 'Ubuntu' 
plt.rcParams['font.monospace'] = 'Ubuntu Mono' 
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams['axes.labelweight'] = 'bold' 
plt.rcParams['axes.titlesize'] = 12 
plt.rcParams['xtick.labelsize'] = 12 
plt.rcParams['ytick.labelsize'] = 12 
plt.rcParams['legend.fontsize'] = 12 
plt.rcParams['figure.titlesize'] = 12 
plt.rcParams['image.cmap'] = 'jet' 
plt.rcParams['image.interpolation'] = 'none' 
plt.rcParams['figure.figsize'] = (12, 10) 
plt.rcParams['axes.grid']=True
plt.rcParams['lines.linewidth'] = 2 
plt.rcParams['lines.markersize'] = 8
colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
'xkcd:scarlet']
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
    'SeatCapacity'
]

top = { "Airline": 100,
        'Destination': 100,
        'FlightType': 10,
        'Sector': 10
        }

columns = ['SeatCapacity']

input_data = raw_data[raw_attributes_used]
transformer = hdp.DataTransformer(top=top, dummy_encode=True)
transformer.fit(input_data)

X_transformed_df = transformer.transform(input_data, columns)


# Be careful for actual passengers = 0
X_df = X_transformed_df[raw_data.LoadFactor != 0]
load_factor = raw_data[raw_data.LoadFactor != 0].LoadFactor
# seat_capacity = raw_data[raw_data.LoadFactor != 0].SeatCapacity
# y_df = pd.DataFrame(data=load_factor.values *
#                     seat_capacity.values, columns=['Number of Passengers'])
# y = y_df.values.ravel()


X = X_df.values

y = load_factor.values.ravel()

n, p = X.shape

# import im_models

# mod_regr = im_models.PredictionModels(cv_outer=10, cv_inner=5)
# mod_regr.fit(X,y)

# y_hat = mod_regr.predict(X)

dataset = raw_data.loc[raw_data.LoadFactor > 0]
training_length = int(2/3*dataset.shape[0])
train_dataset = dataset[0:training_length]
test_dataset = dataset[training_length:]
y_test = test_dataset[["LoadFactor"]]
X_test = test_dataset.drop(['LoadFactor'], axis=1)


trnsf = spreproc.SamplingTransformer(sample_by='hours')
X, _, y = trnsf.fit(train_dataset)

#X2, _, y2 = trnsf.fit(test_dataset)

#schedule = trnsf.from_sampled_to_schedule_format(y2[-300:], X_test[-300:])



#=================KNN - No Shuffle===================#





# K_outer = 10
# K_inner = 5
# CV_outer = KFold(n_splits=K_outer, shuffle=False)

# errors = np.zeros(K_outer)
# errors_rmse = np.zeros(K_outer)
# best_estimators = []
# for i, (train_set, test_set) in enumerate(CV_outer.split(X, y)):
#     X_train = X[train_set]
#     y_train = y[train_set]
#     X_test = X[test_set]
#     y_test = y[test_set]
#     param_grid = {
#         'ccp_alpha': np.linspace(0.4, 0.6, 10),
#         'max_depth': [7, 8, 9],
#         'min_samples_leaf': [14, 15, 16],
#     }
#     regr = GridSearchCV(DecisionTreeRegressor(),
#                         param_grid=param_grid, cv=K_inner, n_jobs=-1, verbose=2)
#     regr.fit(X_train, y_train)
#     best_estimators.append(regr.best_estimator_)
#     y_hat = regr.predict(X_test)
#     error = Helpers.ErrorMetrics.accuracy_per_flight(y_test, y_hat)
#     errors[i] = (np.abs(error)).mean()
#     errors_rmse[i] = Helpers.ErrorMetrics.rmse(y_test, y_hat)

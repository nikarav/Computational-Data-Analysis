from cgitb import enable
from dython.nominal import associations
from pydoc import Helper
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit
import sklearn.preprocessing as preproc
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from sklearn import neighbors 
import matplotlib.pyplot as plt
import seaborn as sns
from importlib.resources import path
import Helpers.ErrorMetrics
import Helpers.DataPreprocessing as hdp
import Helpers.SamplingPreprocessing as spreproc
import Helpers.DataPreprocessing2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
import os
import sys
import Helpers.CrossValidation as CV
import os
import sys
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

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


dataset = raw_data.loc[raw_data.LoadFactor > 0]
dataset = dataset.sort_values(by='ScheduleTime')
throw_away_due_to_covid = len(dataset.loc[(
    raw_data.ScheduleTime.dt.year == 2021) & (raw_data.ScheduleTime.dt.month < 5)].index)

dataset = dataset[throw_away_due_to_covid:]
dataset.drop(['FlightNumber',  'Sector'], axis=1, inplace=True)


transformer = Helpers.DataPreprocessing2.DataTransformer(
    top_percent=80, dummy_encode=True)
transformer.fit(dataset)
data = transformer.transform(dataset)
#data.drop(['Year'], axis=1, inplace=True)
data2, enc_map2 = transformer.target_encode_column_df(
    data, "LoadFactor", 'Destination')

#associations(data2)

data2.corr()
data2.drop(['Month', 'Day', 'Year'], axis = 1, inplace=True)

#data2 = pd.get_dummies(data2, columns=['Destination'])

X_df = data2.drop(['LoadFactor'], axis=1)
y_df = data2['LoadFactor']



# scaler = StandardScaler().fit(X_df)
# features = scaler.transform(X_df)
# X_df[:] = features


X = X_df.values
y = y_df.values.ravel()



n, p = X.shape




#=======================================================#
#=======================================================#
#======================= Tests =========================#
#=======================================================#
#=======================================================#



cv = CV.CrossValidation(cv_inner=5, cv_outer=10)

import im_models
param_instance = im_models.paramsNestedCV()

#================= KNN - Shuffle ===================#

params_knn = param_instance.knn_grid_params()

knn = neighbors.KNeighborsRegressor()
#knn_result = cv.normalCV(X=X, y=y, model=knn, param=params_knn,
#                                name='plots/Knn/knn_all_features.pdf', X_df=X_df)
#knn_result.to_csv(os.path.join(script_path, 'knn_result_shuffle.csv'))



#================= RandomForest - Shuffle ===================#

params_rf = param_instance.random_forest_grid_params()

rf_model = RandomForestRegressor()
#random_forest_res = cv.normalCV(X=X, y=y, model=rf_model, param=params_rf,
#                                name='plots/RandomForest/rf_all_features.pdf', X_df=X_df)
#random_forest_res.to_csv(os.path.join(script_path, 'tests/rf_all_feat.csv'))




#================= ElasticNet - Shuffle ===================#


params_elastic = param_instance.elastic_grid_params()

elastic_model = ElasticNet()
#random_forest_res = cv.normalCV(X=X, y=y, model=elastic_model, param=params_elastic,
#                                name='plots/ElasticNet/Elastic_all_features.pdf', X_df=X_df)
#random_forest_res.to_csv(os.path.join(script_path, 'elastic_all_feat.csv'))



#================= XGBoost - Shuffle ===================#


param_xgb = param_instance.XGBoost_grid_params()

xgb_model = XGBRegressor()
#xgb_forest_res = cv.normalCV(X=X, y=y, model=xgb_model, param=param_xgb,
#                               name='plots/XGBoost/XGBoost_all_feat.pdf', X_df=X_df)
#xgb_forest_res.to_csv(os.path.join(script_path, 'tests/xgboost_all_feat.csv'))



#================= LGBMRegressor - Shuffle ===================#


param_lgbm = param_instance.LGBM_grid_params()

lgbm_model = LGBMRegressor()
#lgbm_res = cv.normalCV(X=X, y=y, model=lgbm_model, param=param_lgbm,
#                               name='plots/LGBM/LGBM_all_feat.pdf', X_df=X_df)
#lgbm_res.to_csv(os.path.join(script_path, 'tests/lgbm.csv'))



#================= SVR - Shuffle ===================#
sv_model = SVR()
param_test={'C':[0.0001,0.001,0.01],
            'kernel':('linear','rbf','poly'),
            'degree':[3,4],
            'gamma':[1,5,10]}
#svr_res = cv.normalCV(X=X, y=y, model=sv_model, param=param_test,
#                                name='plots/RandomForest/svr_all_features.pdf', X_df=X_df)
#svr_res.to_csv(os.path.join(script_path, 'svr_all_feat.csv'))


#================= Logistic Regresion - Shuffle ===================#
logReg_model = LogisticRegression()
param_test ={'C':[10,30,40,45, 50],
              'penalty':('l1','l2')}
#log_reg_res = cv.normalCV(X=X, y=y, model=logReg_model, param=param_test,
#                                name='plots/RandomForest/logReg_all_features.pdf', X_df=X_df)
#log_reg_res.to_csv(os.path.join(script_path, 'LogReg_all_feat.csv'))



#================= Gradient Boosting - Shuffle ===================#
gradient_boost_model = GradientBoostingRegressor()
gradient_param = param_instance.gradient_boosting_grid_params()

#gradient_boost_res = cv.normalCV(X=X, y=y, model=gradient_boost_model, param=gradient_param,
#                                name='plots/RandomForest/gradBoost_all_features.pdf', X_df=X_df)
#gradient_boost_res.to_csv(os.path.join(script_path, 'tests/gradBoost_v2.csv'))



#================= Ridge Regression - Shuffle ===================#
ridge_boost_model = Ridge()
ridge_param = param_instance.ridge_grid_params()

#ridge_res = cv.normalCV(X=X, y=y, model=ridge_boost_model, param=ridge_param,
#                                name='plots/Ridge/ridge_all_features.pdf', X_df=X_df)
#ridge_res.to_csv(os.path.join(script_path, 'ridge_all_feat.csv'))


models = {
    'XGBRBoost': XGBRegressor(
                                max_depth=105, 
                                n_estimators= 100, 
                                learning_rate=0.1, 
                                colsample_bytree=0.9,
                                min_child_weight=25,
                                subsample=0.95
                ),  
    'LGBMBoost': LGBMRegressor(
                                max_bin=190,
                                n_jobs=-1,
                                num_leaves=800,
                                learning_rate=0.15,
                                max_depth=28
                ),
    'AdaBoost':  AdaBoostRegressor(
                        DecisionTreeRegressor( 
                                max_depth=21), 
                                n_estimators=100, 
                                learning_rate=0.24
                ),
    'RandomForest': RandomForestRegressor(
                                max_depth=40,
                                n_estimators= 100
                )
}


#acc_per_model, w_acc_per_model = cv.modelSelection(X=X, y=y, shuffle=True, models=models)
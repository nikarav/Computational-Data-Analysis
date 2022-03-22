from matplotlib.pyplot import grid
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


class PredictionModels():
    def __init__(self, cv_inner=10, cv_outer=10):
        self.cv_inner = cv_inner
        self.cv_outer = cv_outer

    def fit(self, X, y):
        X = X.copy()
        y = y.copy()

        CV = KFold(n_splits=self.cv_outer, shuffle=True)
        self.estimators = []
        for i, (train_index, test_index) in enumerate(CV.split(X, y)):
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            grid_params = {
                # "max_depth": [10],
                # "max_features": ['auto'],
                # 'n_estimators': [10]
            }

            regr = GridSearchCV(RandomForestRegressor(),
                                cv=self.cv_outer, param_grid=grid_params, n_jobs=-1)
            regr.fit(X_train, y_train)
            self.estimators.append(regr.best_estimator_)
            print(f"{i+1} out of {self.cv_outer}")
        print('Done fitting')

    def predict(self, X):
        y_hat_ens = np.zeros((len(self.estimators), X.shape[0]))
        for i, estimator in enumerate(self.estimators):
            y_hat = estimator.predict(X)
            y_hat_ens[i, :] = y_hat.ravel()
        return y_hat_ens.mean()



class paramsNestedCV():
    def __init__(self):
        pass
    
    def knn_grid_params(self):
        return {'n_neighbors':list(range(2,15))
        }
        # return {'n_neighbors':list(range(2,15)),
        #         'weights':('uniform','distance'),
        #         'algorithm':('auto', 'ball_tree', 'kd_tree'),
        #         'p':(1,2)
        #         }

    def random_forest_grid_params(self):
        # return {
        #         'bootstrap': [True],
        #         'max_features': ['auto', 'sqrt'],
        #         'max_depth': [80, 90, 100, 110],
        #         'min_samples_leaf': [3, 4, 5],
        #         'min_samples_split': [8, 10, 12],
        #         'n_estimators': [100, 200, 300, 1000],
        #         'ccp_alpha': np.linspace(0.4, 0.6, 10)
        #         }
        return {
                'max_depth': [30, 40],
                'n_estimators': [500],
                'n_jobs': [-1]
        }
    
    def elastic_grid_params(self):
        return {'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
                'l1_ratio': np.arange(0.01, 1, 0.01)
        }

    def XGBoost_grid_params(self):
        return {'max_depth':[9, 10],
                'n_jobs': [-1],
                'n_estimators':[ 200, 300],
                'min_child_weight':[4,5],
                'learning_rate': [0.01,0.02,0.03,0.04]
        }
    def LGBM_grid_params(self):
        return {
                'max_bin':[190],
                'n_jobs': [-1],
                'num_leaves': [800],
                'learning_rate': [ 0.15],
                'max_depth': [28]
        }

    def gradient_boosting_grid_params(self):
        return {
                'learning_rate': [0.1],
                'subsample'    : [0.9],
                'n_estimators' : [300],
                'max_depth'    : [9]
        }
    
    def ridge_grid_params(self):
        return {
            'alpha': np.arange(0, 1, 0.01)
        }
                
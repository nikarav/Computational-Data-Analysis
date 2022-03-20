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



class KnnNestedCV():
    def __init__(self) -> None:
        pass
    
    def grid_params():
        return {'n_neighbors':[2,3,4,5,6,7,8,9]}

    #def trainCV():


    #def predict():
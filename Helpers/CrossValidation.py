import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

import im_models
import Helpers

class CrossValidation():
    def __init__(self, cv_inner=10, cv_outer=10):
        self.cv_inner = cv_inner
        self.cv_outer = cv_outer
        

    def NestedCV(self, X, y, shuffle):
        CV_outer = KFold(n_splits=self.cv_outer, shuffle=shuffle)

        #errors = np.zeros(K_outer)
        #errors_rmse = np.zeros(K_outer)
        best_estimators = []
        self.estimators = []
        for i, (train_set, test_set) in enumerate(CV_outer.split(X, y)):
            X_train = X[train_set]
            y_train = y[train_set]
            X_test = X[test_set]
            y_test = y[test_set]
            # param_grid = {
            #     'ccp_alpha': np.linspace(0.4, 0.6, 10),
            #     'max_depth': [7, 8, 9],
            #     'min_samples_leaf': [14, 15, 16],
            # }

            param = im_models.KnnNestedCV().grid_params()
            regr = GridSearchCV(KNeighborsRegressor(),
                                param_grid=param, cv=self.cv_inner, n_jobs=-1, verbose=2)
            regr.fit(X_train, y_train)
            self.estimators.append(regr.best_estimator_)
            y_hat = regr.predict(X_test)
            error = Helpers.ErrorMetrics.accuracy_per_flight(y_test, y_hat)
            #errors[i] = (np.abs(error)).mean()
            #errors_rmse[i] = Helpers.ErrorMetrics.rmse(y_test, y_hat)

    def predict(self, X):
        y_hat_ens = np.zeros((len(self.estimators), X.shape[0]))
        for i, estimator in enumerate(self.estimators):
            y_hat = estimator.predict(X)
            y_hat_ens[i, :] = y_hat.ravel()
        return y_hat_ens.mean()


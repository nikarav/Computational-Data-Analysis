import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from  sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats as st

from sklearn.metrics import make_scorer

import scoring.accuracy as accuracy

def ttest_onemodel(y_true, yhat, loss_norm_p=1, alpha=0.05):    
    # perform statistical comparison of the models
    # compute z with squared error.

    zA = 1 - np.abs((y_true - yhat)/y_true) ** loss_norm_p
    CI = st.t.interval(1 - alpha, df=len(zA) - 1, loc=np.mean(zA), scale=st.sem(zA))
    return np.mean(zA), CI

class CrossValidation():
    def __init__(self, cv_inner=5, cv_outer=10):
        self.cv_inner = cv_inner
        self.cv_outer = cv_outer
        


    def NestedCV(self, X, y, shuffle, model, param):
        score = accuracy.scoring()

        CV_outer = KFold(n_splits=self.cv_outer, shuffle=shuffle, random_state=42)
        
        outer_results = []
        #errors = np.zeros(K_outer)
        #errors_rmse = np.zeros(K_outer)
        best_estimators = []
        self.estimators = []
        self.bestParam = []
 
        for i, (train_set, test_set) in enumerate(CV_outer.split(X, y)):
            X_train = X[train_set]
            y_train = y[train_set]
            X_test = X[test_set]
            y_test = y[test_set]


            
            # Define custom scoring evaluation for GridSearchCV
            accuracy_flight = make_scorer(score.accuracy, greater_is_better=True)

            # define search
            search = GridSearchCV(estimator=model,
                                param_grid=param,
                                scoring=accuracy_flight,
                                cv=self.cv_inner, 
                                n_jobs=-1, 
                                verbose=2
                                )
            # execute search
            result = search.fit(X_train, y_train)

            # get the best performing model fit on the whole training set
            best_model = result.best_estimator_ 

            self.estimators.append(result.best_estimator_)
            self.bestParam.append(result.best_params_)
            self.featureImportance.append()
            # evaluate model on the hold out dataset
            y_hat = best_model.predict(X_test)
            
            # We want len(y_test)
            ytest_len = len(y_test)
            y_len = len(y)
            y_diff = (ytest_len/y_len) 

            # evaluate the model

            acc = score.accuracy(y_test, y_hat)
            outer_results.append(acc)
            # report progress
            print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

            # evaluate model on the hold out dataset
            #y_hat = regr.predict(X_test)
            #error = Helpers.ErrorMetrics.accuracy_per_flight(y_test, y_hat)
            #errors[i] = (np.abs(error)).mean()
            #errors_rmse[i] = Helpers.ErrorMetrics.rmse(y_test, y_hat)
        print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))
        return pd.DataFrame({'Best Estimator':self.estimators, 'Best Param': self.bestParam, 'Accuracy': outer_results })

    def modelSelection(self, X, y, shuffle, models):
        score = accuracy.scoring()

        CV_outer = KFold(n_splits=self.cv_outer, shuffle=shuffle, random_state=42)
        y_len = len(y)

        self.w_accuracies = np.zeros((10,4))

        for i, (train_set, test_set) in enumerate(CV_outer.split(X, y)):
            X_train = X[train_set]
            y_train = y[train_set]
            X_test = X[test_set]
            y_test = y[test_set]
            
            ytest_len = len(y_test) 
            y_diff = (ytest_len/y_len)

            for j,(name, model) in enumerate(models.items()):
                # Train each model
                result = model.fit(X_train, y_train)

                # evaluate each model on the hold out dataset
                y_hat = result.predict(X_test)
                acc = score.accuracy(y_test, y_hat)
                w_acc = acc * y_diff

                self.w_accuracies[i,j] = w_acc
                self.accuracies[i,j] = acc
                print('>acc=%.3f, w_acc=%s' % (acc, w_acc))
                print('=======================================')
        
        return self.accuracies, self.w_accuracies



    def predict(self, X):
        y_hat_ens = np.zeros((len(self.estimators), X.shape[0]))
        for i, estimator in enumerate(self.estimators):
            y_hat = estimator.predict(X)
            y_hat_ens[i, :] = y_hat.ravel()
        return y_hat_ens.mean()

    def normalCV(self, X, y, model, param, name, X_df):
        score = accuracy.scoring()

        outer_results = []
        #errors = np.zeros(K_outer)
        #errors_rmse = np.zeros(K_outer)
        best_estimators = []
        self.estimators = []
        self.bestParam = []
       
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Define custom scoring evaluation for GridSearchCV
        accuracy_flight = make_scorer(score.accuracy, greater_is_better=True)

        # define search
        search = GridSearchCV(estimator=model,
                                param_grid=param,
                                #scoring=accuracy_flight,
                                scoring='neg_mean_absolute_error',
                                cv=5, 
                                n_jobs=-1, 
                                verbose=10
                                )

        # execute search
        result = search.fit(X_train, y_train)

        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_ 

        self.estimators.append(result.best_estimator_)
        self.bestParam.append(result.best_params_)
        # evaluate model on the hold out dataset
        y_hat = best_model.predict(X_test)

        # evaluate the model
        acc = score.accuracy(y_test, y_hat)

        outer_results.append(acc)
        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

            # evaluate model on the hold out dataset
            #y_hat = regr.predict(X_test)
            #error = Helpers.ErrorMetrics.accuracy_per_flight(y_test, y_hat)
            #errors[i] = (np.abs(error)).mean()
            #errors_rmse[i] = Helpers.ErrorMetrics.rmse(y_test, y_hat)
        print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))
        
        # importances = result.best_estimator_.feature_importances_
        # #
        # # Sort the feature importance in descending order
        # #
        # sorted_indices = np.argsort(importances)[::-1]


        # importance_dictionary = dict(
        #         zip(np.asarray(X_df.columns), result.best_estimator_.feature_importances_))
        # importance_dictionary = dict(
        #         sorted(importance_dictionary.items(), key=lambda x: x[1], reverse=True))


        # print(importance_dictionary)

        # plt.title('Feature Importance')
        # plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
        # plt.xticks(range(X_df.shape[1]), X_df.columns[sorted_indices], rotation=90)
        # plt.tight_layout()
        # plt.savefig(name)
        return pd.DataFrame({'Best Estimator':self.estimators, 'Best Param': self.bestParam, 'Accuracy': outer_results })

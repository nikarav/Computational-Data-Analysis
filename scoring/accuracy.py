from sklearn.metrics import fbeta_score, make_scorer
import numpy as np

class scoring():
    def __init__(self, X):
        self.X = X

    def accuracy(self, y_true, y_pred):
        return np.mean(1 - np.abs((y_true - y_pred) / y_true))
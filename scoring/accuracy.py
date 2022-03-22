from sklearn.metrics import fbeta_score, make_scorer
import numpy as np

class scoring():
    def __init__(self):
        pass

    def accuracy(self, y_true, y_pred):
        return np.mean(1 - np.abs((y_true - y_pred) / y_true))
    def accuracy2(self, y_true, y_pred):
        return np.abs((y_true - y_pred) / y_true)
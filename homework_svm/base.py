from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator


class BaseVectorMethod(BaseEstimator, ABC):

    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @staticmethod
    def accuracy(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return float(np.mean(y_true == y_pred))

    @staticmethod
    def precision(y_true, y_pred, positive_label=1):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
        fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
        denom = tp + fp
        return 0.0 if denom == 0 else float(tp / denom)

    @staticmethod
    def recall(y_true, y_pred, positive_label=1):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
        fn = np.sum((y_pred != positive_label) & (y_true == positive_label))
        denom = tp + fn
        return 0.0 if denom == 0 else float(tp / denom)

    @staticmethod
    def f1(y_true, y_pred, positive_label=1):
        p = BaseVectorMethod.precision(y_true, y_pred, positive_label)
        r = BaseVectorMethod.recall(y_true, y_pred, positive_label)
        denom = p + r
        return 0.0 if denom == 0 else float(2 * p * r / denom)

    @staticmethod
    def mse(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mae(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def r2(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

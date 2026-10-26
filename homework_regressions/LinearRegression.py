import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.weight_matrix = None

    def fit(self, fit_data, y, eta=1e-4, max_iters=10000, eps=0.01):
        n_observations = fit_data.shape[0]
        n_features = fit_data.shape[1]

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        weight_matrix = np.zeros((n_features + 1, 1))
        aug_data = np.column_stack([fit_data, np.ones(n_observations)])
        for i in range(max_iters):
            predictors = aug_data @ weight_matrix
            grad = 2 * aug_data.T @ (predictors - y)
            if np.linalg.norm(grad) < eps:
                break
            weight_matrix = weight_matrix - grad * eta
        self.weight_matrix = weight_matrix
        self.n_features_in_ = n_features
        return self

    def predict(self, predict_data):
        n_observations = predict_data.shape[0]
        aug_data = np.column_stack([predict_data, np.ones(n_observations)])
        predictors = aug_data @ self.weight_matrix
        return predictors.ravel()

    def score(self, X, y):
        """R²."""
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    def metrics(self, X, y):
        """Базовые метрики регрессии: mse, mae, r2."""
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        mse = float(np.mean((y - y_pred) ** 2))
        mae = float(np.mean(np.abs(y - y_pred)))
        r2 = self.score(X, y)
        return {"mse": mse, "mae": mae, "r2": r2}

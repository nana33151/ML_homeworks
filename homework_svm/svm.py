
import numpy as np

from base import BaseVectorMethod


class SVM(BaseVectorMethod):

    def __init__(self, C=1.0, epsilon=0.1):
        self.C = C
        self.epsilon = epsilon
        self.task_ = None
        self.w_ = None
        self.b_ = None
        self.classes_ = None

    def _fit_binary(self, X, y_pm1, eta, max_iters, tol):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0

        for t in range(max_iters):
            margins = y_pm1 * (X @ w + b)
            mask = margins < 1.0
            if np.any(mask):
                grad_w = w - (self.C / n_samples) * (y_pm1[mask] @ X[mask])
                grad_b = -(self.C / n_samples) * np.sum(y_pm1[mask])
            else:
                grad_w = w.copy()
                grad_b = 0.0

            if np.linalg.norm(grad_w) + abs(grad_b) < tol:
                break

            step = eta / np.sqrt(t + 1)
            w = w - step * grad_w
            b = b - step * grad_b

        return w, b

    def _fit_regression(self, X, y, eta, max_iters, tol):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0

        for t in range(max_iters):
            residual = X @ w + b - y
            mask = np.abs(residual) > self.epsilon
            if np.any(mask):
                signs = np.sign(residual[mask])
                grad_w = w + (self.C / n_samples) * (signs @ X[mask])
                grad_b = (self.C / n_samples) * np.sum(signs)
            else:
                grad_w = w.copy()
                grad_b = 0.0

            if np.linalg.norm(grad_w) + abs(grad_b) < tol:
                break

            step = eta / np.sqrt(t + 1)
            w = w - step * grad_w
            b = b - step * grad_b

        return w, b

    def fit(self, X, y, task="classification", eta=0.1, max_iters=2000, tol=1e-4):
        if task not in ("classification", "regression"):
            raise ValueError("task должен быть 'classification' или 'regression'")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.task_ = task
        self.n_features_in_ = X.shape[1]

        if task == "regression":
            self.w_, self.b_ = self._fit_regression(X, y.astype(float), eta, max_iters, tol)
            self.classes_ = None
            return self


        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("нужно минимум 2 класса")

        W = np.zeros((X.shape[1], n_classes))
        B = np.zeros(n_classes)
        for k, cls in enumerate(self.classes_):
            y_pm1 = np.where(y == cls, 1.0, -1.0)
            W[:, k], B[k] = self._fit_binary(X, y_pm1, eta, max_iters, tol)

        self.w_ = W
        self.b_ = B
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w_ + self.b_

    def predict(self, X):
        scores = self.decision_function(X)
        if self.task_ == "regression":
            return scores
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        if self.task_ == "regression":
            return self.r2(y, y_pred)
        return self.accuracy(y, y_pred)

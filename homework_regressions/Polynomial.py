from LinearRegression import LinearRegression
import itertools
import numpy as np


class Polynomial(LinearRegression):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree

    def get_polynomial_combinations(self, X, n_features, max_degree):
        feature_indices = list(range(n_features))
        all_combinations = []

        for d in range(0, max_degree + 1):
            step_combinations = list(
                itertools.combinations_with_replacement(feature_indices, d)
            )
            all_combinations.extend(step_combinations)

        X_poly_list = []
        for comb in all_combinations:
            if len(comb) == 0:
                new_feature = np.ones((X.shape[0], 1))
            else:
                sub_matrix = X[:, comb]
                new_feature = np.prod(sub_matrix, axis=1).reshape(-1, 1)
            X_poly_list.append(new_feature)

        return np.hstack(X_poly_list)

    def fit(self, fit_data, y, eta=0.01, max_iters=10000, eps=0.01):
        aug_data = self.get_polynomial_combinations(
            fit_data, fit_data.shape[1], self.degree
        )
        n_observations = aug_data.shape[0]
        n_features = aug_data.shape[1]

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.weight_matrix = np.zeros((n_features, 1))

        for i in range(max_iters):
            predictors = aug_data @ self.weight_matrix
            grad = (2 / n_observations) * aug_data.T @ (predictors - y)
            if np.linalg.norm(grad) < eps:
                break
            self.weight_matrix = self.weight_matrix - grad * eta
        self.n_features_in_ = fit_data.shape[1]
        return self

    def predict(self, predict_data):
        aug_data = self.get_polynomial_combinations(
            predict_data, predict_data.shape[1], self.degree
        )
        predictors = aug_data @ self.weight_matrix
        return predictors.ravel()

from LinearRegression import LinearRegression
import itertools
import numpy as np


class Polynomial(LinearRegression):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        self.grid_scores_ = None 

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

    def fit_grid(
        self,
        X,
        y,
        degrees=(1, 2, 3, 4, 5),
        val_size=0.25,
        random_state=42,
        eta=0.01,
        max_iters=10000,
        eps=0.01,
    ):
       
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n = X.shape[0]
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_size))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        scores = {}
        best_degree, best_r2 = None, -np.inf
        for d in degrees:
            candidate = Polynomial(degree=d)
            candidate.fit(X_train, y_train, eta=eta, max_iters=max_iters, eps=eps)
            r2 = candidate.score(X_val, y_val)
            scores[d] = r2
            if r2 > best_r2:
                best_degree, best_r2 = d, r2

        self.degree = best_degree
        self.grid_scores_ = scores
        return self.fit(X, y, eta=eta, max_iters=max_iters, eps=eps)

    def predict(self, predict_data):
        aug_data = self.get_polynomial_combinations(
            predict_data, predict_data.shape[1], self.degree
        )
        predictors = aug_data @ self.weight_matrix
        return predictors.ravel()

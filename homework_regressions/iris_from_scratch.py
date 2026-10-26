import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.weight_matrix = None

    def fit(self, fit_data, fit_labels, eta=0.1, max_iters=200, eps=0.1):
        n_observations = fit_data.shape[0]
        n_features = fit_data.shape[1]
        n_labels = len(np.unique(fit_labels))

        weight_matrix = np.column_stack(
            [np.zeros((n_features + 1, 1)) for i in range(n_labels - 1)]
        )
        aug_data = np.column_stack([fit_data, np.ones(n_observations)])
        for i in range(max_iters):
            predictors = aug_data @ weight_matrix
            full_predictors = np.hstack([predictors, np.zeros((n_observations, 1))])
            probabilities = softmax(full_predictors, axis=1)

            mask = (fit_labels[:, None] == np.arange(n_labels)).astype(int)
            grad = 1 / n_observations * aug_data.T @ (probabilities - mask)
            grad = grad[:, : n_labels - 1]
            if np.linalg.norm(grad) < eps:
                break
            weight_matrix = weight_matrix - eta * grad

        self.weight_matrix = weight_matrix
        self.n_features_in_ = n_features
        self.classes_ = np.unique(fit_labels)
        return self

    def predict(self, predict_data):
        n_observations = predict_data.shape[0]
        aug_data = np.column_stack([predict_data, np.ones(n_observations)])
        predictors = aug_data @ self.weight_matrix
        full_predictors = np.hstack([predictors, np.zeros((n_observations, 1))])
        probabilities = softmax(full_predictors, axis=1)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        """Accuracy."""
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    def metrics(self, X, y):
        """Базовые метрики: accuracy, precision_macro, recall_macro."""
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        classes = np.unique(y)
        precisions, recalls = [], []
        for c in classes:
            tp = np.sum((y_pred == c) & (y == c))
            fp = np.sum((y_pred == c) & (y != c))
            fn = np.sum((y_pred != c) & (y == c))
            precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
            recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
        return {
            "accuracy": self.score(X, y),
            "precision_macro": float(np.mean(precisions)),
            "recall_macro": float(np.mean(recalls)),
        }



if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train, eta=0.5, max_iters=500, eps=1e-6)
    print(f"Точность: {model.score(X_test_scaled, y_test):.4f}")
    print(model.metrics(X_test_scaled, y_test))

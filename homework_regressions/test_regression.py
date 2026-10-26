"""Тесты линейной и полиномиальной регрессии на базовом датасете."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from LinearRegression import LinearRegression
from Polynomial import Polynomial


def _scaled_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def test_linear_regression():
    X, y = make_regression(
        n_samples=200, n_features=3, noise=5.0, random_state=42
    )
    X_train, X_test, y_train, y_test = _scaled_split(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train, eta=1e-4, max_iters=20000, eps=1e-3)

    m = model.metrics(X_test, y_test)
    assert set(m) == {"mse", "mae", "r2"}
    assert m["r2"] > 0.9, f"ожидали R² > 0.9, получили {m['r2']}"
    assert m["mse"] >= 0 and m["mae"] >= 0


def test_polynomial_regression():
    rng = np.random.RandomState(42)
    X = rng.uniform(-1.5, 1.5, size=(250, 1))
    y = 1.5 * X[:, 0] ** 2 - 0.5 * X[:, 0] + 0.2 + rng.normal(0, 0.05, size=250)

    X_train, X_test, y_train, y_test = _scaled_split(X, y)

    model = Polynomial(degree=2)
    model.fit(X_train, y_train, eta=0.05, max_iters=20000, eps=1e-4)

    m = model.metrics(X_test, y_test)
    assert set(m) == {"mse", "mae", "r2"}
    assert m["r2"] > 0.9, f"ожидали R² > 0.9, получили {m['r2']}"


def test_polynomial_degree_grid():
    rng = np.random.RandomState(0)
    X = rng.uniform(-1.5, 1.5, size=(300, 1))
    y = 1.5 * X[:, 0] ** 2 - 0.5 * X[:, 0] + 0.2 + rng.normal(0, 0.05, size=300)

    X_train, X_test, y_train, y_test = _scaled_split(X, y)

    model = Polynomial()
    model.fit_grid(
        X_train,
        y_train,
        degrees=(1, 2, 3, 4),
        eta=0.05,
        max_iters=15000,
        eps=1e-4,
    )

    assert model.grid_scores_ is not None
    assert set(model.grid_scores_) == {1, 2, 3, 4}
    assert model.degree >= 2  # линейная степень хуже квадратичной
    assert model.metrics(X_test, y_test)["r2"] > 0.9


if __name__ == "__main__":
    test_linear_regression()
    test_polynomial_regression()
    test_polynomial_degree_grid()
    print("Все тесты прошли.")

import time

import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import transmep.optimize as optimize
from transmep import get_device


def test_grid_search_kr_rbf_cpu():
    alphas = torch.logspace(-1, 0, 2)
    gammas = torch.logspace(-2, -1, 2)
    validation_iterations = 1000
    x, y = make_regression(n_features=10, noise=0.1)

    training_size = int(0.9 * len(x))

    prior_time = time.time()
    best_alpha, best_gamma, scores_torch = optimize.grid_search_kr_rbf_fork(
        torch.tensor(x), torch.tensor(y), alphas, gammas, validation_iterations, training_size
    )
    scores_torch = torch.mean(scores_torch, dim=-1)
    time_torch = time.time() - prior_time

    time_sklearn = _check_grid_search_output(
        alphas,
        best_alpha,
        best_gamma,
        gammas,
        scores_torch,
        training_size,
        validation_iterations,
        x,
        y,
    )

    print("torch runtime: %.4fs\nsklearn runtime: %.4fs" % (time_torch, time_sklearn))


def test_grid_search_kr_rbf_batched():
    alphas = torch.logspace(-1, 0, 2)
    gammas = torch.logspace(-2, -1, 2)
    validation_iterations = 1000
    x, y = make_regression(n_features=10, noise=0.1)

    training_size = int(0.9 * len(x))

    prior_time = time.time()
    best_alpha, best_gamma, scores_torch = optimize.grid_search_kr_rbf_batched(
        torch.tensor(x),
        torch.tensor(y),
        alphas,
        gammas,
        validation_iterations,
        training_size,
        9,
        7,
        get_device(),
    )
    scores_torch = scores_torch.cpu()
    scores_torch = torch.mean(scores_torch, dim=-1)
    time_torch = time.time() - prior_time

    time_sklearn = _check_grid_search_output(
        alphas,
        best_alpha,
        best_gamma,
        gammas,
        scores_torch,
        training_size,
        validation_iterations,
        x,
        y,
    )

    print("torch runtime: %.4fs\nsklearn runtime: %.4fs" % (time_torch, time_sklearn))


def _check_grid_search_output(
    alphas, best_alpha, best_gamma, gammas, scores_torch, training_size, validation_iterations, x, y
):
    assert scores_torch[
        list(alphas).index(best_alpha), list(gammas).index(best_gamma)
    ] == torch.min(scores_torch)
    prior_time = time.time()
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            score_sklearn = _score_kr_rbf_sklearn(
                x, y, validation_iterations, training_size, float(alpha), float(gamma)
            )
            score_torch = float(scores_torch[i, j])
            assert np.isclose(score_sklearn, score_torch, atol=0.05), (
                "High deviation between sklearn baseline and torch solution: %.4f vs %.4f (%.4e)"
                % (score_sklearn, score_torch, score_sklearn - score_torch)
            )
    time_sklearn = time.time() - prior_time
    return time_sklearn


def _score_kr_rbf_sklearn(
    x: np.ndarray,
    y: np.ndarray,
    validation_iterations: int,
    training_size: int,
    alpha: float,
    gamma: float,
) -> float:
    y = (y - np.mean(y)) / np.std(y, ddof=1)
    scores = []
    for _ in range(validation_iterations):
        train_x, val_x, train_y, val_y = train_test_split(x, y, train_size=training_size)
        ridge = KernelRidge(kernel="rbf", gamma=gamma / train_x.shape[1], alpha=alpha)
        ridge.fit(train_x, train_y)
        prediction = ridge.predict(val_x)
        scores.append(mean_squared_error(val_y, prediction))
    return float(np.mean(scores))

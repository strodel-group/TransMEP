import time

import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import transmep.optimize as optimize
from transmep import get_device


def test_grid_search_kr_rbf_cpu() -> None:
    """
    Test that the grid search is implemented correctly by comparing to scikit-learn.
    This tests the forking implementation on the CPU.

    :return: None
    """

    # Example setup
    alphas = torch.logspace(-1, 0, 2)
    gammas = torch.logspace(-2, -1, 2)
    validation_iterations = 1000
    x, y = make_regression(n_features=10, noise=0.1)
    training_size = int(0.9 * len(x))

    # Perform grid search using forking implementation
    prior_time = time.time()
    best_alpha, best_gamma, scores_torch = optimize.grid_search_kr_rbf_fork(
        torch.tensor(x), torch.tensor(y), alphas, gammas, validation_iterations, training_size
    )
    scores_torch = torch.mean(scores_torch, dim=-1)
    time_torch = time.time() - prior_time

    # Check with the scikit-learn reference
    time_sklearn = _check_grid_search_output(
        alphas,
        gammas,
        best_alpha,
        best_gamma,
        scores_torch,
        training_size,
        validation_iterations,
        x,
        y,
    )

    print("torch runtime: %.4fs\nsklearn runtime: %.4fs" % (time_torch, time_sklearn))


def test_grid_search_kr_rbf_batched():
    """
    Test that the grid search is implemented correctly by comparing to scikit-learn.
    This tests the batched implementation on the GPU.

    :return: None
    """

    # Example setup
    alphas = torch.logspace(-1, 0, 2)
    gammas = torch.logspace(-2, -1, 2)
    validation_iterations = 1000
    x, y = make_regression(n_features=10, noise=0.1)
    training_size = int(0.9 * len(x))

    # Perform grid search using batched implementation
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

    # Check with the scikit-learn reference
    time_sklearn = _check_grid_search_output(
        alphas,
        gammas,
        best_alpha,
        best_gamma,
        scores_torch,
        training_size,
        validation_iterations,
        x,
        y,
    )

    print("torch runtime: %.4fs\nsklearn runtime: %.4fs" % (time_torch, time_sklearn))


def _check_grid_search_output(
    alphas: torch.Tensor,
    gammas: torch.Tensor,
    best_alpha: float,
    best_gamma: float,
    scores_torch: torch.Tensor,
    training_size: int,
    validation_iterations: int,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Checks given grid search results by comparing them to scikit-learn.

    :param alphas: Alpha values tested
    :param gammas: Gamma values tested
    :param best_alpha: Best alpha value (checked)
    :param best_gamma: Best gamma value (checked)
    :param scores_torch: Scores of all alpha, gamma pairs (checked)
    :param training_size: Size of the training set
    :param validation_iterations: Number of validation iterations for scikit-learn
    :param x: Data input
    :param y: Data target values
    :return: Wall time for scikit-learn
    """
    # Check that best alpha and gamma values are selected correctly
    assert scores_torch[
        list(alphas).index(best_alpha), list(gammas).index(best_gamma)
    ] == torch.min(scores_torch)

    # Check all scores by comparing them to scikit-learn
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
    """
    Fit a scikit-learn kernel ridge regression to the given data with iterative shuffle-split.

    :param x: Data input.
    :param y: Data target values.
    :param validation_iterations: Iterations of shuffle-split.
    :param training_size: Size of the training dataset.
    :param alpha: Alpha hyperparameter value
    :param gamma: Gamma hyperparameter value
    :return: Mean MSE
    """
    # Normalize target values (scikit-learn does not do this automatically)
    y = (y - np.mean(y)) / np.std(y, ddof=1)

    # Iterative shuffle-split
    scores = []
    for _ in range(validation_iterations):
        train_x, val_x, train_y, val_y = train_test_split(x, y, train_size=training_size)
        ridge = KernelRidge(kernel="rbf", gamma=gamma / train_x.shape[1], alpha=alpha)
        ridge.fit(train_x, train_y)
        prediction = ridge.predict(val_x)
        scores.append(mean_squared_error(val_y, prediction))
    return float(np.mean(scores))

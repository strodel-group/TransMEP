import gc
import math
from typing import Tuple

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from transmep import compute_pairwise_distances

# This file contains the routines for optimized grid search.
# Every grid search iterates over a grid of alpha and gamma values and estimates each pair's
# generalization error (MSE) by repeated holdout (shuffle-split).
# Note: Because we ignore the estimated standard deviation during grid search these routines
# are identical for kernel ridge regression and Gaussian Process regression.
# Thus, these routines use the more simple kernel ridge regression formulation.
# There are two implementations: Forking and batched.
# The forking implementation is more suited for CPUs and parallelizes over alpha, gamma pairs,
# while the batched implementation is more suited for GPUs and parallelizes over validation
# iterations.


@torch.jit.script
def _score_kr_rbf_fork(
    distances: torch.Tensor,
    y: torch.Tensor,
    gamma: torch.Tensor,
    alpha: torch.Tensor,
    validation_iterations: int,
    training_size: int,
) -> torch.Tensor:
    """
    Sub-routine of the forking implementation.
    This method estimates the generalization error of a given alpha, gamma pair.

    :param distances: Precomputed pairwise distances.
    :param y: Target values.
    :param gamma: Gamma value.
    :param alpha: Alpha value.
    :param validation_iterations: Number of repeated holdout (shuffle-split) iterations.
    :param training_size: Size of the training dataset in each split.
    :return: Estimated generalization errors of all validation iterations.
    """

    # Precompute kernel matrix
    n = distances.shape[0]
    kernel_matrix = torch.exp(-gamma * distances)

    # Repeated holdout
    scores = torch.zeros(validation_iterations)
    for i in range(validation_iterations):

        # Split data
        permutation = torch.randperm(n, device=kernel_matrix.device)
        train_idx = permutation[:training_size]
        val_idx = permutation[training_size:]
        train_kernel = kernel_matrix[train_idx, :][:, train_idx]
        train_y = y[train_idx]
        val_kernel = kernel_matrix[val_idx, :][:, train_idx]
        val_y = y[val_idx]

        # Fit model on training dataset
        train_kernel = train_kernel + alpha * torch.eye(train_kernel.shape[0])
        coefficients = torch.cholesky_solve(
            train_y.unsqueeze(1), torch.linalg.cholesky(train_kernel)
        )[:, 0]

        # Score on validation dataset
        prediction = torch.matmul(val_kernel, coefficients)
        mean_squared_error = F.mse_loss(prediction, val_y)
        scores[i] = mean_squared_error

    return scores


@torch.jit.script
def grid_search_kr_rbf_fork(
    training_embeddings: torch.Tensor,
    y: torch.Tensor,
    alphas: torch.Tensor,
    gammas: torch.Tensor,
    validation_iterations: int,
    training_size: int,
) -> Tuple[float, float, torch.Tensor]:
    """
    Main routine for forking grid search implementation.
    This implementation is better suited for usage on the CPU.
    It uses TorchScript and its forking methods for multithreading.

    :param training_embeddings: Embeddings of the training data.
    :param y: Target values.
    :param alphas: Alpha values to test.
    :param gammas: Gamma values to test.
    :param validation_iterations: Number of repeated holdout (shuffle-split) iterations.
    :param training_size: Size of the training dataset in each split.
    :return: Best alpha value, best gamma value, and all computed scores.
    """

    # Normalize y because we assume a mean of zero and a standard deviation of one
    intercept = torch.mean(y)
    scale = torch.std(y)
    y = (y - intercept) / scale

    # Precompute distance matrix
    # This cannot use transmep.compute_pairwise_distances() because of TorchScript
    distances = (
        torch.cdist(
            training_embeddings.unsqueeze(0),
            training_embeddings.unsqueeze(0),
            compute_mode="donot_use_mm_for_euclid_dist",
        )[0]
        ** 2
        / training_embeddings.shape[1]
    )

    # Parallelized grid search
    futures = [
        [
            torch.jit.fork(
                _score_kr_rbf_fork,
                distances,
                y,
                gamma,
                alpha,
                validation_iterations,
                training_size,
            )
            for gamma in gammas
        ]
        for alpha in alphas
    ]

    # Await all futures
    scores = torch.stack(
        [torch.stack([torch.jit.wait(future) for future in futures_row]) for futures_row in futures]
    )

    # Find best score
    argmin = torch.argmin(torch.mean(scores, dim=-1))  # returns index in flat array
    best_alpha = alphas[torch.div(argmin, len(gammas), rounding_mode="trunc")]
    best_gamma = gammas[argmin % len(gammas)]
    return best_alpha, best_gamma, scores


def grid_search_kr_rbf_batched(
    x: torch.Tensor,
    y: torch.Tensor,
    alphas: torch.Tensor,
    gammas: torch.Tensor,
    validation_iterations: int,
    training_size: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
    show_progress_bar: bool = True,
    precomputed_distances: bool = False,
) -> Tuple[float, float, torch.Tensor]:
    """
    Main routine for batched grid search implementation.
    This implementation is better suited for usage on the GPU.

    :param x: Input data.
    :param y: Target values.
    :param alphas: Alpha values to test.
    :param gammas: Gamma values to test.
    :param validation_iterations: Number of repeated holdout (shuffle-split) iterations.
    :param training_size: Size of the training dataset in each split.
    :param batch_size: Batch of validation iterations size.
    :param block_size: Block size during pairwise distance calculation.
    :param device: Device to perform computations on.
    :param show_progress_bar: Whether to show a progress bar.
    :param precomputed_distances: If False, the input data x is interpreted as a tensor with
        embeddings of shape samples x embedding_dim. Else, the input data is interpreted as
        a square matrix of pairwise distances.
    :return: The best alpha value, the best gamma value, and all computed scores.
    """

    with torch.no_grad():

        # Move small tensors to device
        y = y.to(device)
        alphas = alphas.to(device)
        gammas = gammas.to(device)

        # Normalize y because we assume a mean of zero and a standard deviation of one.
        intercept = torch.mean(y)
        scale = torch.std(y)
        y = (y - intercept) / scale

        # Precompute distance matrix
        if precomputed_distances:
            distances = x.to(device)
        else:
            distances = compute_pairwise_distances(x, block_size=block_size, device=device)

        # Do validation iterations in batches
        scores = torch.zeros((len(alphas), len(gammas), validation_iterations), device=x.device)
        iterator = range(0, validation_iterations, batch_size)
        if show_progress_bar:
            # Add a progress bar if enabled
            iterator = tqdm(iterator, desc="HPO")
        for batch_start in iterator:
            batch_end = min(validation_iterations, batch_start + batch_size)

            # Pre-select holdout sets and store them in contiguous format in memory
            permutations = torch.stack(
                [torch.randperm(x.shape[0], device=device) for _ in range(batch_end - batch_start)]
            )
            train_idx = permutations[:, :training_size]
            val_idx = permutations[:, training_size:]
            train_distances = distances[train_idx[:, :, None], train_idx[:, None]].contiguous()
            train_y = y[train_idx].contiguous()
            val_distances = distances[val_idx[:, :, None], train_idx[:, None]].contiguous()
            val_y = y[val_idx].contiguous()

            # Perform grid search
            eye = torch.eye(training_size, device=device).unsqueeze(0)
            for j, gamma in enumerate(gammas):

                # Kernel matrix can be precomputed for a given gamma
                train_kernel = torch.exp(-gamma * train_distances)
                val_kernel = torch.exp(-gamma * val_distances)

                for i, alpha in enumerate(alphas):

                    # Final training kernel matrix with regularization
                    regularized_kernel_matrix = train_kernel + eye * alpha

                    # Fit model
                    cholesky, info = torch.linalg.cholesky_ex(
                        regularized_kernel_matrix, check_errors=False
                    )
                    if torch.count_nonzero(info) > 0:
                        # There was an error during Cholesky decomposition
                        scores[i, j, batch_start:batch_end] = math.nan
                    else:
                        # Extract coefficients
                        coefficients = torch.cholesky_solve(train_y[:, :, None], cholesky)[:, :, 0]

                        # Score on validation dataset
                        prediction = torch.matmul(val_kernel, coefficients[:, :, None])[:, :, 0]
                        mean_squared_error = torch.mean((prediction - val_y) ** 2, dim=1)
                        scores[i, j, batch_start:batch_end] = mean_squared_error

            # Cleanup after each iteration (avoids OOM errors)
            del (
                permutations,
                train_idx,
                val_idx,
                train_distances,
                train_y,
                val_distances,
                val_y,
                train_kernel,
                val_kernel,
                cholesky,
                coefficients,
                prediction,
            )
            gc.collect()
            torch.cuda.empty_cache()

        # Extract min value respecting NaNs
        mean_scores = torch.mean(scores, dim=-1)
        if torch.all(torch.isnan(mean_scores)):
            raise RuntimeError("All scores are NaN!")
        else:
            mean_scores = torch.nan_to_num(mean_scores, nan=torch.finfo(mean_scores.dtype).max)
            argmin = torch.argmin(mean_scores)  # returns index in flat array
            best_alpha = float(alphas[torch.div(argmin, len(gammas), rounding_mode="trunc")])
            best_gamma = float(gammas[argmin % len(gammas)])
            return best_alpha, best_gamma, scores

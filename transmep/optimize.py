import gc
import math
from typing import Tuple

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from transmep import compute_pairwise_distances


@torch.jit.script
def _score_kr_rbf_fork(
    distances: torch.Tensor,
    y: torch.Tensor,
    gamma: torch.Tensor,
    alpha: torch.Tensor,
    validation_iterations: int,
    training_size: int,
) -> torch.Tensor:

    # precompute kernel matrix
    n = distances.shape[0]
    kernel_matrix = torch.exp(-gamma * distances)

    # repeated holdout
    scores = torch.zeros(validation_iterations)
    for i in range(validation_iterations):
        # split data
        permutation = torch.randperm(n, device=kernel_matrix.device)
        train_idx = permutation[:training_size]
        val_idx = permutation[training_size:]
        train_kernel = kernel_matrix[train_idx, :][:, train_idx]
        train_y = y[train_idx]
        val_kernel = kernel_matrix[val_idx, :][:, train_idx]
        val_y = y[val_idx]

        # fit model on training dataset
        train_kernel = train_kernel + alpha * torch.eye(train_kernel.shape[0])
        coefficients = torch.cholesky_solve(
            train_y.unsqueeze(1), torch.linalg.cholesky(train_kernel)
        )[:, 0]

        # score on validation dataset
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

    # normalize y because we assume a mean of zero
    intercept = torch.mean(y)
    scale = torch.std(y)
    y = (y - intercept) / scale

    # precompute distance matrix
    distances = (
        torch.cdist(
            training_embeddings.unsqueeze(0),
            training_embeddings.unsqueeze(0),
            compute_mode="donot_use_mm_for_euclid_dist",
        )[0]
        ** 2
        / training_embeddings.shape[1]
    )

    # parallelized grid search
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
    scores = torch.stack(
        [torch.stack([torch.jit.wait(future) for future in futures_row]) for futures_row in futures]
    )
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

    with torch.no_grad():

        # move small tensors to device
        y = y.to(device)
        alphas = alphas.to(device)
        gammas = gammas.to(device)

        # normalize y because we assume a mean of zero
        intercept = torch.mean(y)
        scale = torch.std(y)
        y = (y - intercept) / scale

        # precompute distance matrix
        if precomputed_distances:
            distances = x.to(device)
        else:
            distances = compute_pairwise_distances(x, block_size=block_size, device=device)

        # do validation iterations in batches
        scores = torch.zeros((len(alphas), len(gammas), validation_iterations), device=x.device)

        iterator = range(0, validation_iterations, batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="HPO")
        for batch_start in iterator:
            batch_end = min(validation_iterations, batch_start + batch_size)

            # pre-select holdout sets
            permutations = torch.stack(
                [torch.randperm(x.shape[0], device=device) for _ in range(batch_end - batch_start)]
            )
            train_idx = permutations[:, :training_size]
            val_idx = permutations[:, training_size:]
            train_distances = distances[train_idx[:, :, None], train_idx[:, None]].contiguous()
            train_y = y[train_idx].contiguous()
            val_distances = distances[val_idx[:, :, None], train_idx[:, None]].contiguous()
            val_y = y[val_idx].contiguous()

            # perform grid search
            eye = torch.eye(training_size, device=device).unsqueeze(0)
            for j, gamma in enumerate(gammas):
                train_kernel = torch.exp(-gamma * train_distances)
                val_kernel = torch.exp(-gamma * val_distances)
                for i, alpha in enumerate(alphas):

                    # fit model on training dataset
                    cholesky, info = torch.linalg.cholesky_ex(
                        train_kernel + eye * alpha, check_errors=False
                    )
                    if torch.count_nonzero(info) > 0:
                        scores[i, j, batch_start:batch_end] = math.nan
                    else:
                        coefficients = torch.cholesky_solve(train_y[:, :, None], cholesky)[:, :, 0]

                        # score on validation dataset
                        prediction = torch.matmul(val_kernel, coefficients[:, :, None])[:, :, 0]
                        mean_squared_error = torch.mean((prediction - val_y) ** 2, dim=1)
                        scores[i, j, batch_start:batch_end] = mean_squared_error

            # cleanup
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

        # extract min value respecting NaNs
        mean_scores = torch.mean(scores, dim=-1)
        if torch.all(torch.isnan(mean_scores)):
            raise RuntimeError("All scores are NaN!")
        else:
            mean_scores = torch.nan_to_num(mean_scores, nan=torch.finfo(mean_scores.dtype).max)
            argmin = torch.argmin(mean_scores)  # returns index in flat array
            best_alpha = float(alphas[torch.div(argmin, len(gammas), rounding_mode="trunc")])
            best_gamma = float(gammas[argmin % len(gammas)])
            return best_alpha, best_gamma, scores

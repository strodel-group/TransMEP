from typing import Tuple, Union

import esm
import fsspec
import torch
from torch import nn

from transmep import compute_pairwise_distances
from transmep.foundation_model import load_foundation_model


class Model(nn.Module):
    def __init__(
        self,
        transformer_name: str,
        transformer: Union[esm.ProteinBertModel, esm.ESM2] = None,
    ):
        super(Model, self).__init__()
        self.transformer_name = transformer_name
        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer, _ = load_foundation_model(transformer_name)
        self.alpha = None
        self.gamma = None
        self.train_embeddings = None
        self.intercept = None
        self.scale = None
        self.train_kernel_cholesky = None
        self.dual_coefficients = None

    def fit(
        self,
        alpha: float,
        gamma: float,
        train_embeddings: torch.Tensor,
        train_y: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        block_size: int = None,
    ):
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.train_embeddings = nn.Parameter(train_embeddings, requires_grad=False)

        # precompute what can be precomputed
        self.intercept = nn.Parameter(torch.mean(train_y), requires_grad=False)
        self.scale = nn.Parameter(torch.std(train_y), requires_grad=False)
        train_y = (train_y - self.intercept) / self.scale
        train_kernel = _compute_kernel(
            self.train_embeddings.data,
            self.train_embeddings.data,
            self.gamma.data,
            device=device,
            block_size=block_size,
        ) + self.alpha.to(device) * torch.eye(self.train_embeddings.shape[0], device=device)
        self.train_kernel_cholesky = nn.Parameter(
            torch.linalg.cholesky(train_kernel).cpu(), requires_grad=False
        )
        self.dual_coefficients = nn.Parameter(
            torch.cholesky_solve(train_y.unsqueeze(1), self.train_kernel_cholesky)[:, 0].cpu(),
            requires_grad=False,
        )

    def forward(
        self, x: torch.Tensor, embedded: bool = False, return_std: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # compute kernel matrix
        if embedded:
            embeddings = x
        else:
            embeddings = self._embed(x)
        kernel = _compute_kernel(
            embeddings, self.train_embeddings.data, self.gamma.data, device=embeddings.device
        )

        # compute mean
        mean = torch.matmul(kernel, self.dual_coefficients) * self.scale + self.intercept

        # if requested, compute standard deviation estimate
        if return_std:
            tmp = torch.linalg.solve_triangular(self.train_kernel_cholesky, kernel.T, upper=False)
            diagonal = torch.sum(tmp**2, dim=0)
            std = torch.sqrt(1 - diagonal) * self.scale
            return mean, std
        else:
            return mean

    def compute_importance(self, x_tokens: torch.Tensor) -> torch.Tensor:
        # compute kernel matrix
        embeddings = self._embed(x_tokens)
        kernel = _compute_kernel(
            embeddings, self.train_embeddings.data, self.gamma.data, device=embeddings.device
        )

        # compute importance value
        weights_abs = torch.abs(kernel * self.dual_coefficients.unsqueeze(0))
        importance = weights_abs / torch.sum(weights_abs, dim=1, keepdim=True)
        return importance

    def _embed(self, x_tokens: torch.Tensor) -> torch.Tensor:
        plm = self.transformer(x_tokens, repr_layers=[self.transformer.num_layers])
        embeddings = torch.reshape(
            plm["representations"][self.transformer.num_layers][:, 1:-1], (x_tokens.shape[0], -1)
        )
        return embeddings


def _compute_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    block_size: int = None,
) -> torch.Tensor:
    distances = compute_pairwise_distances(x1, x2, device=device, block_size=block_size)
    return torch.exp(-gamma.to(device) * distances)


def save_model(model: Model, file: str):
    state_dict = model.state_dict()
    state_dict["transformer_name"] = model.transformer_name
    for key in list(state_dict.keys()):
        if key.startswith("transformer."):
            del state_dict[key]
    with fsspec.open(file, "wb") as fd:
        torch.save(state_dict, fd)


def load_model(file: str):
    with fsspec.open(file, "rb") as fd:
        state_dict = torch.load(fd)
    transformer, alphabet = load_foundation_model(state_dict["transformer_name"])
    model = Model(state_dict["transformer_name"], transformer=transformer)
    model.alpha = nn.Parameter(state_dict["alpha"], requires_grad=False)
    model.gamma = nn.Parameter(state_dict["gamma"], requires_grad=False)
    model.train_embeddings = nn.Parameter(state_dict["train_embeddings"], requires_grad=False)
    model.intercept = nn.Parameter(state_dict["intercept"], requires_grad=False)
    model.scale = nn.Parameter(state_dict["scale"], requires_grad=False)
    model.train_kernel_cholesky = nn.Parameter(
        state_dict["train_kernel_cholesky"], requires_grad=False
    )
    model.dual_coefficients = nn.Parameter(state_dict["dual_coefficients"], requires_grad=False)
    return model, alphabet

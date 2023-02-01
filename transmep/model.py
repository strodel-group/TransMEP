from typing import Tuple, Union

import esm
import fsspec
import torch
from torch import nn

from transmep import compute_pairwise_distances
from transmep.foundation_model import load_foundation_model

# Main model implementation for inference.


class Model(nn.Module):
    """
    Torch Gaussian Process model with foundation model included.
    """

    def __init__(
        self,
        transformer_name: str,
        transformer: Union[esm.ProteinBertModel, esm.ESM2] = None,
    ):
        """
        Create a new model given the foundation model.
        This model is not yet fitted to any data.

        :param transformer_name: ID of the foundation model.
        :param transformer: Foundation model object. If None, the foundation model is loaded.
        """
        super(Model, self).__init__()
        self.transformer_name = transformer_name
        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer, _ = load_foundation_model(transformer_name)

        # Initialized during fit()
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
    ) -> None:
        # Save values that we need later
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.train_embeddings = nn.Parameter(train_embeddings, requires_grad=False)

        # Precompute what can be precomputed (cf. to SI)
        self.intercept = nn.Parameter(torch.mean(train_y), requires_grad=False)
        self.scale = nn.Parameter(torch.std(train_y), requires_grad=False)
        train_y = (train_y - self.intercept) / self.scale
        train_kernel = _compute_kernel(
            self.train_embeddings.data,
            self.train_embeddings.data,
            self.gamma.data,
            device=device,
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
        """
        Apply inference on the given input sequences.

        :param x: Input sequences.
        :param embedded: If True, x is expected to contain the embeddings of shape
            batch x embedding_dim. Else, x should contain the tokens of the input sequence
            of shape batch x sequence.
        :param return_std: Whether to compute an estimate of the standard deviation.
        :return: The predicted mean and optionally the estimated standard deviation.
        """
        # Compute embeddings if necessary
        if embedded:
            embeddings = x
        else:
            embeddings = self._embed(x)

        # Compute kernel matrix of query and training samples
        kernel = _compute_kernel(
            embeddings, self.train_embeddings.data, self.gamma.data, device=embeddings.device
        )

        # Compute mean (cf. to SI)
        mean = torch.matmul(kernel, self.dual_coefficients) * self.scale + self.intercept

        # If requested, compute standard deviation estimate (cf. to SI)
        if return_std:
            tmp = torch.linalg.solve_triangular(self.train_kernel_cholesky, kernel.T, upper=False)
            diagonal = torch.sum(tmp**2, dim=0)
            std = torch.sqrt(1 - diagonal) * self.scale
            return mean, std
        else:
            return mean

    def compute_importance(self, x: torch.Tensor, embedded: bool = False) -> torch.Tensor:
        """
        Estimate training samples importance.

        :param x: Input sequences.
        :param embedded: If True, x is expected to contain the embeddings of shape
            batch x embedding_dim. Else, x should contain the tokens of the input sequence
            of shape batch x sequence.
        :return: Importance values of shape len(training_embeddings).
        """
        # Compute embeddings if necessary
        if embedded:
            embeddings = x
        else:
            embeddings = self._embed(x)

        # Compute kernel matrix of query and training samples
        kernel = _compute_kernel(
            embeddings, self.train_embeddings.data, self.gamma.data, device=embeddings.device
        )

        # Compute importance value (cf. to SI)
        weights_abs = torch.abs(kernel * self.dual_coefficients.unsqueeze(0))
        importance = weights_abs / torch.sum(weights_abs, dim=1, keepdim=True)
        return importance

    def _embed(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        Pass the given tokens through the foundation model.

        :param x_tokens: Tokenized version of the input sequence.
        :return: Embedding.
        """
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
) -> torch.Tensor:
    """
    Compute kernel matrix between x1 and x2.

    :param x1: Rows of the kernel matrix.
    :param x2: Columns of the kernel matrix.
    :param gamma: Gamma value.
    :param device: Device to use for computation.
    :return: The kernel matrix.
    """
    distances = compute_pairwise_distances(x1, x2, device=device)
    return torch.exp(-gamma.to(device) * distances)


def save_model(model: Model, file: str) -> None:
    """
    Save a model.
    The foundation model's weights are not saved, but only its ID.

    :param model: Model to save.
    :param file: Path to save to. Supports remote paths via fsspec.
    :return: None
    """
    state_dict = model.state_dict()

    # Only keep the foundation model ID, not its weights.
    state_dict["transformer_name"] = model.transformer_name
    for key in list(state_dict.keys()):
        if key.startswith("transformer."):
            del state_dict[key]

    # Save
    with fsspec.open(file, "wb") as fd:
        torch.save(state_dict, fd)


def load_model(file: str) -> Tuple[Model, esm.Alphabet]:
    """
    Load a model.

    :param file: The path to the saved model. Supports remote paths via fsspec.
    :return: The model and its alphabet.
    """

    # Load the state dict
    with fsspec.open(file, "rb") as fd:
        state_dict = torch.load(fd)

    # Load the foundation model by its ID
    transformer, alphabet = load_foundation_model(state_dict["transformer_name"])

    # Create the model
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

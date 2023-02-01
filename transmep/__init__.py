from typing import Callable, Union

import esm
import torch
from tqdm.auto import tqdm

from transmep.data import Dataset

# This file contains some utility functions.


def get_device() -> torch.device:
    """
    Select a torch device from the environment.
    Prefers cuda:0 if available.

    :return: A torch device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def embed(
    dataset: Dataset,
    foundation_model: Union[esm.ESM2, esm.ProteinBertModel],
    batch_converter: Callable,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Embed the given dataset using the given foundation model.
    :param dataset: The dataset to process.
    :param foundation_model: Foundation model.
    :param batch_converter: The batch converter (tokenizer) of the foundation model.
    :param batch_size: Batch size to use for embedding.
    :return: The embeddings of shape len(dataset) x embedding_dim.
    """

    # Preparations
    device = get_device()
    foundation_model.eval().to(device)
    layer_index = foundation_model.num_layers

    # Embed dataset in batches
    embeddings = None  # lazily initialized, because embedding_dim varies
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Embedding"):
            batch_end = min(batch_start + batch_size, len(dataset))

            # Apply foundation model
            _, _, batch_tokens = batch_converter(
                [(str(i), dataset.variants[i]) for i in range(batch_start, batch_end)]
            )
            result = foundation_model(batch_tokens.to(device), repr_layers=[layer_index])

            # Extract embeddings from the last layer
            embedding = result["representations"][layer_index][:, 1:-1].cpu()

            # Lazily initialize embeddings array
            if embeddings is None:
                embeddings = torch.zeros((len(dataset),) + embedding.shape[1:], dtype=torch.float)
            embeddings[batch_start:batch_end] = embedding

    # Remove transformer from GPU memory
    foundation_model.cpu()

    # Reshape embeddings by flattening sequence dimension
    embeddings = torch.reshape(embeddings, (embeddings.shape[0], -1))
    return embeddings


def compute_pairwise_distances(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor = None,
    block_size: int = None,
    device: torch.device = torch.device("cpu"),
    enable_progress_bar: bool = False,
) -> torch.Tensor:
    """
    Compute pairwise distances between two embedding lists.
    Supports operation in blocks to reduce GPU memory usage.
    The pairwise distance is the squared L2-norm divided by the embedding dimensionality.
    Note: This is different from the regular L2 norm, but better suited for the RBF kernel.

    :param embeddings_a: List of embeddings A.
    :param embeddings_b: List of embeddings B. If None, it is assumed to be A.
        This allows for some optimizations due to symmetries.
    :param block_size: Block size. Set to None to process everything in one shot.
    :param device: Device to use. Defaults to CPU.
    :param enable_progress_bar: Whether to show a progress bar.
    :return: The pairwise distances matrix.
    """

    # Determine whether we have the symmetric case.
    if embeddings_b is None:
        embeddings_b = embeddings_a
        symmetric = True
    else:
        symmetric = False

    # Short path if no block-building is required
    if block_size is None:
        return (
            torch.cdist(
                embeddings_a.unsqueeze(0).to(device),
                embeddings_b.unsqueeze(0).to(device),
                compute_mode="donot_use_mm_for_euclid_dist",
            )[0]
            ** 2
            / embeddings_a.shape[1]
        )

    # Create empty pairwise distances matrix
    n_a = len(embeddings_a)
    n_b = len(embeddings_b)
    distances = torch.zeros((n_a, n_b), dtype=embeddings_a.dtype, device=device)

    # Prepare blocks iterator
    blocks = []
    for block_start_a in range(0, n_a, block_size):
        block_end_a = min(block_start_a + block_size, n_a)
        for block_start_b in range(block_start_a if symmetric else 0, n_b, block_size):
            block_end_b = min(block_start_b + block_size, n_b)
            blocks.append((block_start_a, block_end_a, block_start_b, block_end_b))

    # Wrap the iterator with a progress bar if enabled
    if enable_progress_bar:
        iterator = tqdm(blocks, desc="Computing distance matrix")
    else:
        iterator = iter(blocks)

    # Iterate over the blocks
    for (block_start_a, block_end_a, block_start_b, block_end_b) in iterator:
        # Compute distances for block
        block = torch.cdist(
            embeddings_a[block_start_a:block_end_a].unsqueeze(0).to(device),
            embeddings_b[block_start_b:block_end_b].unsqueeze(0).to(device),
            compute_mode="donot_use_mm_for_euclid_dist",  # not stable enough numerically
        )[0]

        # Compute scaled L2-norm
        block = block**2 / embeddings_a.shape[1]

        # Assign block to corresponding positions
        distances[block_start_a:block_end_a, block_start_b:block_end_b] = block
        if symmetric:
            distances[block_start_b:block_end_b, block_start_a:block_end_a] = block.transpose(0, 1)

    return distances

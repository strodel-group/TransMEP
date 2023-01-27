from typing import Callable, Union

import esm
import numpy as np
import torch
from tqdm.auto import tqdm

from transmep.data import Dataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def embed(
    dataset: Dataset,
    transformer: Union[esm.ESM2, esm.ProteinBertModel],
    batch_converter: Callable,
    batch_size: int = 1,
) -> torch.Tensor:
    device = get_device()
    transformer.eval().to(device)
    layer_index = transformer.num_layers

    # Embed dataset
    embeddings = None
    with torch.no_grad():
        # for i, entry in enumerate(tqdm(dataset.variants, desc="Embedding")):
        for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Embedding"):
            batch_end = min(batch_start + batch_size, len(dataset))
            _, _, batch_tokens = batch_converter(
                [(str(i), dataset.variants[i]) for i in range(batch_start, batch_end)]
            )
            result = transformer(batch_tokens.to(device), repr_layers=[layer_index])
            embedding = result["representations"][layer_index][:, 1:-1].cpu()
            if embeddings is None:
                embeddings = torch.zeros((len(dataset),) + embedding.shape[1:], dtype=torch.float)
            embeddings[batch_start:batch_end] = embedding

    # Remove transformer from GPU memory
    transformer.cpu()

    # Reshape embeddings
    embeddings = torch.reshape(embeddings, (embeddings.shape[0], -1))
    return embeddings


def compute_pairwise_distances(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor = None,
    block_size: int = None,
    device: torch.device = torch.device("cpu"),
    out_np: np.ndarray = None,
    enable_progress_bar: bool = False,
) -> torch.Tensor:
    if embeddings_b is None:
        embeddings_b = embeddings_a
        symmetric = True
    else:
        symmetric = False

    # short path if no block-building is required
    if block_size is None:
        assert out_np is None
        return (
            torch.cdist(
                embeddings_a.unsqueeze(0).to(device),
                embeddings_b.unsqueeze(0).to(device),
                compute_mode="donot_use_mm_for_euclid_dist",
            )[0]
            ** 2
            / embeddings_a.shape[1]
        )

    # compute distances in blocks
    n_a = len(embeddings_a)
    n_b = len(embeddings_b)
    if out_np is None:
        distances = torch.zeros((n_a, n_b), dtype=embeddings_a.dtype, device=device)
    else:
        distances = out_np

    # prepare blocks iterator
    blocks = []
    for block_start_a in range(0, n_a, block_size):
        block_end_a = min(block_start_a + block_size, n_a)
        for block_start_b in range(block_start_a if symmetric else 0, n_b, block_size):
            block_end_b = min(block_start_b + block_size, n_b)
            blocks.append((block_start_a, block_end_a, block_start_b, block_end_b))
    if enable_progress_bar:
        iterator = tqdm(blocks, desc="Computing distance matrix")
    else:
        iterator = iter(blocks)

    # iterate over the blocks
    for (block_start_a, block_end_a, block_start_b, block_end_b) in iterator:
        # compute distances for block
        block = torch.cdist(
            embeddings_a[block_start_a:block_end_a].unsqueeze(0).to(device),
            embeddings_b[block_start_b:block_end_b].unsqueeze(0).to(device),
            compute_mode="donot_use_mm_for_euclid_dist",
        )[0]

        # compute scaled L2-norm
        block = block**2 / embeddings_a.shape[1]

        # assign block to corresponding positions
        if out_np is not None:
            block = block.cpu().numpy()
        distances[block_start_a:block_end_a, block_start_b:block_end_b] = block
        if symmetric:
            if out_np is None:
                distances[block_start_b:block_end_b, block_start_a:block_end_a] = block.transpose(
                    0, 1
                )
            else:
                distances[block_start_b:block_end_b, block_start_a:block_end_a] = block.T

    return distances

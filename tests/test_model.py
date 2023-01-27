import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from transmep import embed, get_device
from transmep.data import load_dataset
from transmep.foundation_model import load_foundation_model
from transmep.model import Model


def test_model():
    train = load_dataset(
        "https://mutation-prediction.thoffbauer.de/transmep/c-wt.txt",
        "https://mutation-prediction.thoffbauer.de/transmep/c-variants-train.csv",
    )
    test = load_dataset(
        "https://mutation-prediction.thoffbauer.de/transmep/c-wt.txt",
        "https://mutation-prediction.thoffbauer.de/transmep/c-variants-train.csv",
    )
    alpha = 0.1
    gamma = 1
    foundation_model = "esm2_t30_150M_UR50D"
    transformer, alphabet = load_foundation_model(foundation_model)
    batch_converter = alphabet.get_batch_converter()
    device = get_device()
    transformer = transformer.eval().to(device)
    train_embeddings = embed(train, transformer, batch_converter).cpu()
    test_embeddings = embed(test, transformer, batch_converter).cpu()

    with torch.no_grad():
        torch_model = Model(foundation_model, transformer=transformer)
        torch_model.fit(alpha, gamma, train_embeddings, torch.tensor(train.y), device=device)
        torch_model = torch_model.to(device)
        assert not torch.isnan(torch_model.dual_coefficients).any()
        assert not torch.isnan(torch_model.train_kernel_cholesky).any()
        torch_mean = np.zeros((len(test),), dtype=np.float32)
        torch_std = np.zeros((len(test),), dtype=np.float32)
        for i in range(len(test)):
            _, _, batch_tokens = batch_converter([("placeholder", test.variants[i])])
            mean, std = torch_model(batch_tokens.to(device), return_std=True)
            torch_mean[i] = mean.cpu().numpy()[0]
            torch_std[i] = std.cpu().numpy()[0]

    sklearn_mean, sklearn_std = _predict_sklearn(
        train_embeddings.numpy(), train.y, test_embeddings.numpy(), alpha, gamma
    )

    assert np.allclose(torch_mean, sklearn_mean, atol=1e-2)
    assert np.allclose(torch_std, sklearn_std, atol=1e-2)


def _predict_sklearn(train_embeddings, train_y, test_embeddings, alpha, gamma):
    model = GaussianProcessRegressor(
        RBF(
            length_scale=np.sqrt(train_embeddings.shape[1] / (2 * gamma)),
            length_scale_bounds="fixed",
        ),
        normalize_y=True,
        alpha=alpha,
    )
    model.fit(train_embeddings, train_y)
    mean, std = model.predict(test_embeddings, return_std=True)
    return mean, std

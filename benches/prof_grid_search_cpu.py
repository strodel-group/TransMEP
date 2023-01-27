import torch
from sklearn.datasets import make_regression
from torch._C._autograd import ProfilerActivity
from torch.profiler import profile

import transmep.optimize as optimize


def main():
    alphas = torch.logspace(-3, 6, 10)
    gammas = torch.logspace(-3, 6, 10)
    validation_iterations = 100
    x, y = make_regression(n_samples=500, n_features=10, noise=0.1)

    training_size = int(0.9 * len(x))

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        best_alpha, best_gamma, scores_torch = optimize.grid_search_kr_rbf_fork(
            torch.tensor(x), torch.tensor(y), alphas, gammas, validation_iterations, training_size
        )
    print(scores_torch.shape)
    print(prof.key_averages().table(sort_by="cpu_time_total"))
    prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    main()

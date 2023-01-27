import time

import plotly.graph_objs as go
import torch
from plotly.subplots import make_subplots
from sklearn.datasets import make_regression

import transmep.optimize as optimize


def main():
    grid_steps = 25
    fork_cpu = []
    batched_cpu = []
    batched_gpu = []
    iterations = [10, 50, 100, 500, 1000]
    samples = 100
    collect_runtime("fork_cpu", samples, grid_steps, iterations[0])
    collect_runtime("batched_cpu", samples, grid_steps, iterations[0])
    collect_runtime("batched_gpu", samples, grid_steps, iterations[0])
    for validation_iterations in iterations:
        print("Collecting data for validation_iterations = %d" % validation_iterations)
        print("fork_cpu")
        runtime = collect_runtime("fork_cpu", samples, grid_steps, validation_iterations)
        print("%.4f s" % runtime)
        fork_cpu.append(runtime)
        print("batched_cpu")
        runtime = collect_runtime("batched_cpu", samples, grid_steps, validation_iterations)
        print("%.4f s" % runtime)
        batched_cpu.append(runtime)
        print("batched_gpu")
        runtime = collect_runtime("batched_gpu", samples, grid_steps, validation_iterations)
        print("%.4f s" % runtime)
        batched_gpu.append(runtime)

    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=fork_cpu,
            name="Forking CPU",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=batched_cpu,
            name="batched CPU",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=batched_gpu,
            name="batched GPU",
        )
    )
    fig.update_xaxes(type="log", title="validation iterations")
    fig.update_yaxes(type="log", title="runtime")
    fig.show(renderer="browser")


def collect_runtime(type: str, samples: int, grid_steps: int, validation_iterations: int) -> float:
    x, y = make_regression(n_samples=samples, noise=0.1)
    x = torch.tensor(x)
    y = torch.tensor(y)
    training_size = int(0.9 * len(x))
    alphas = torch.logspace(-3, 6, grid_steps)
    gammas = torch.logspace(-3, 6, grid_steps)
    prior_time = time.time()
    if type == "fork_cpu":
        optimize.grid_search_kr_rbf_fork(x, y, alphas, gammas, validation_iterations, training_size)
    elif type == "batched_cpu":
        optimize.grid_search_kr_rbf_batched(
            x, y, alphas, gammas, validation_iterations, training_size
        )
    elif type == "batched_gpu":
        optimize.grid_search_kr_rbf_batched(
            x.cuda(), y.cuda(), alphas.cuda(), gammas.cuda(), validation_iterations, training_size
        )
    return time.time() - prior_time


if __name__ == "__main__":
    main()

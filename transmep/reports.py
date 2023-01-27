import json
from typing import List

import click
import fsspec
import numpy as np
import plotly.graph_objs as go
import torch
from arch.bootstrap import IIDBootstrap
from captum.attr import LayerIntegratedGradients
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
from tqdm import tqdm

from transmep import get_device, stats
from transmep.data import get_mutations, get_mutations_notation, load_dataset
from transmep.model import load_model


def r2(
    base_path: str,
    output_formats: List[str],
    model_path: str,
    wildtype_path: str,
    variants_path: str,
    **_
):
    model, alphabet = load_model(model_path)
    batch_converter = alphabet.get_batch_converter()
    device = get_device()
    model = model.eval().to(device)
    dataset = load_dataset(wildtype_path, variants_path)
    predictions_mean, predictions_std = _make_predictions(batch_converter, dataset, model)

    noise_samples = 1000
    predictions_sampled = predictions_mean[None, :] + np.random.default_rng().normal(
        scale=predictions_std, size=(noise_samples, len(predictions_std))
    )

    def metric(values):
        samples = np.mean(values, axis=0)
        return r2_score(dataset.y, samples)

    lower, _, upper = stats.bootstrap(predictions_sampled, metric=metric)
    estimate = r2_score(dataset.y, predictions_mean)

    if "json" in output_formats:
        with fsspec.open(base_path + "-r2.json", "w") as fd:
            json.dump(
                {
                    "estimate": float(estimate),
                    "confidence_interval": [float(lower), float(upper)],
                },
                fd,
                indent=4,
            )
    if "console" in output_formats:
        click.echo("Estimated R² = %.4f" % estimate)
        click.echo("95%% confidence interval: [%.4f, %.4f]" % (lower, upper))


def mutation_attribution(
    base_path: str,
    output_formats: List[str],
    model_path: str,
    wildtype_path: str,
    variants_path: str,
    attribution_steps: int,
    **_
):
    model, alphabet = load_model(model_path)
    batch_converter = alphabet.get_batch_converter()
    device = get_device()
    model = model.eval().to(device)
    dataset = load_dataset(wildtype_path, variants_path)
    algorithm = LayerIntegratedGradients(model, model.transformer.embed_tokens)
    mutations = get_mutations(dataset)

    _, _, wt_tokens = batch_converter([("placeholder", dataset.wildtype)])
    wt_tokens = wt_tokens.to(device)
    max_error = 0
    attributions = np.zeros((len(dataset), len(mutations)), dtype=np.float32)
    for i in tqdm(range(len(dataset)), desc="Attributing"):
        _, _, batch_tokens = batch_converter([("placeholder", dataset.variants[i])])
        attribution, err = algorithm.attribute(
            batch_tokens.to(device),
            wt_tokens,
            return_convergence_delta=True,
            n_steps=attribution_steps,
            attribute_to_layer_input=False,
            internal_batch_size=1,
        )
        max_error = max(max_error, torch.max(err))
        attribution = torch.sum(attribution[0], dim=1).detach().cpu().numpy()
        for p, (a, b) in enumerate(zip(dataset.wildtype, dataset.variants[i])):
            if a != b:
                j = mutations.index((a, p, b))
                attributions[i, j] = attribution[p + 1]
    click.echo("Maximum error of integral estimation per variant: %.4e" % max_error)

    mutation_names = ["%s%d%s" % (a, p + 1, b) for (a, p, b) in mutations]
    variant_names = [
        get_mutations_notation(dataset.wildtype, variant) for variant in dataset.variants
    ]

    if "json" in output_formats:
        with fsspec.open(base_path + "-mutation-attribution.json", "w") as fd:
            json.dump(
                {
                    "mutations": mutation_names,
                    "variants": variant_names,
                    "attributions": [
                        [float(x) for x in attributions[i]] for i in range(attributions.shape[0])
                    ],
                },
                fd,
                indent=4,
            )
    if _should_plot(output_formats):
        fig = make_subplots()
        fig.add_trace(
            go.Heatmap(
                x=variant_names,
                y=mutation_names,
                z=attributions.T,
                zmid=0,
                showscale=True,
            )
        )
        fig.update_layout(title="Mutation attribution")
        _write_plot(fig, output_formats, base_path, "mutation-attribution")


def grid_search(base_path: str, output_formats: List[str], grid_path: str, **_):
    with fsspec.open(grid_path, "rb") as fd:
        grid = np.load(fd)
        alphas = grid["alphas"]
        gammas = grid["gammas"]
        scores = grid["scores"]
    grid_shape = (len(alphas), len(gammas))

    def metric_mean(scores_sample):
        return np.mean(scores_sample, axis=0).reshape(-1)

    scores_lower, scores_estimate, scores_upper = stats.bootstrap(
        np.transpose(scores, (2, 0, 1)), metric=metric_mean
    )
    scores_lower = scores_lower.reshape(grid_shape)
    scores_estimate = scores_estimate.reshape(grid_shape)
    scores_upper = scores_upper.reshape(grid_shape)
    best_alpha, best_gamma = np.unravel_index(np.argmin(scores_estimate), scores_estimate.shape)

    def metric_diff(scores_sample, best_score_sample):
        diff = (np.mean(scores_sample, axis=0) - np.mean(best_score_sample)).reshape(-1)
        return diff

    diff_bs = IIDBootstrap(np.transpose(scores, (2, 0, 1)), scores[best_alpha, best_gamma])
    diff_lower, _ = diff_bs.conf_int(metric_diff, reps=1000, method="bca", size=0.95, tail="lower")
    diff_lower = diff_lower.reshape(grid_shape)
    similar_to_best = diff_lower <= 0

    if "console" in output_formats:
        print(
            "Best validation score: %.4f at alpha = %.4e, gamma = %.4e"
            % (scores_estimate[best_alpha, best_gamma], alphas[best_alpha], gammas[best_gamma])
        )
        print(
            "95%% confidence interval: [%.4f, %.4f]"
            % (scores_lower[best_alpha, best_gamma], scores_upper[best_alpha, best_gamma])
        )
        print(
            "Number of cells which might be better (95%% confidence): %d (%.2f%%)"
            % (
                np.count_nonzero(similar_to_best),
                np.count_nonzero(similar_to_best) * 100.0 / np.size(similar_to_best),
            )
        )
    if _should_plot(output_formats):
        fig = make_subplots()
        fig.add_trace(
            go.Heatmap(
                x=alphas,
                y=gammas,
                z=np.log10(scores_estimate.T),
                hovertemplate="alpha = %{x:.4e}<br>gamma = %{y:.4e}<br>value = %{text}<extra></extra>",
                text=[
                    [
                        "%.4f [%.4f, %.4f]"
                        % (scores_estimate[j, i], scores_lower[j, i], scores_upper[j, i])
                        for j in range(len(alphas))
                    ]
                    for i in range(len(gammas))
                ],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[alphas[best_alpha]],
                y=[gammas[best_gamma]],
                text="Optimum",
                mode="markers+text",
                textposition="bottom center",
                marker_symbol="star",
                marker_size=15,
                textfont=dict(
                    size=18,
                    color="white",
                ),
                hovertemplate="alpha = %.4e<br>gamma = %.4e<br>value = %.4f [%.4f, %.4f]<extra></extra>"
                % (
                    alphas[best_alpha],
                    gammas[best_gamma],
                    scores_estimate[best_alpha, best_gamma],
                    scores_lower[best_alpha, best_gamma],
                    scores_upper[best_alpha, best_gamma],
                ),
            )
        )
        fig.update_layout(
            showlegend=False,
            title="Grid search",
            xaxis_type="log",
            xaxis_title="alpha",
            yaxis_type="log",
            yaxis_title="gamma",
        )
        _write_plot(fig, output_formats, base_path, "grid-search")
    if "json" in output_formats:
        with fsspec.open(base_path + "-grid-search.json", "w") as fd:
            json.dump(
                {
                    "alphas": [float(x) for x in alphas],
                    "gammas": [float(x) for x in gammas],
                    "scores": {
                        "estimate": [[float(x) for x in row] for row in scores_estimate],
                        "lower": [[float(x) for x in row] for row in scores_lower],
                        "upper": [[float(x) for x in row] for row in scores_upper],
                    },
                    "lower_best_diff": [[float(x) for x in row] for row in diff_lower],
                },
                fd,
                indent=4,
            )


def training_samples_importance(
    base_path: str,
    output_formats: List[str],
    model_path: str,
    wildtype_path: str,
    train_variants_path: str,
    variants_path: str,
    **_
):
    training_dataset = load_dataset(wildtype_path, train_variants_path)
    dataset = load_dataset(wildtype_path, variants_path)
    device = get_device()
    model, alphabet = load_model(model_path)
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    importances = np.zeros((len(dataset), len(training_dataset)))
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Calculating importance"):
            _, _, batch_tokens = batch_converter([("placeholder", dataset.variants[i])])
            importance = model.compute_importance(batch_tokens.to(device))[0].cpu().numpy()
            importances[i] = importance

    train_variant_names = [
        get_mutations_notation(training_dataset.wildtype, variant)
        for variant in training_dataset.variants
    ]
    variant_names = [
        get_mutations_notation(dataset.wildtype, variant) for variant in dataset.variants
    ]

    if "json" in output_formats:
        with fsspec.open(base_path + "-training-samples-importance.json", "w") as fd:
            json.dump(
                {
                    "train_variants": train_variant_names,
                    "variants": variant_names,
                    "importance": [[float(x) for x in row] for row in importances],
                },
                fd,
                indent=4,
            )
    if _should_plot(output_formats):
        fig = make_subplots()
        fig.add_trace(
            go.Heatmap(
                x=train_variant_names,
                y=variant_names,
                z=importances,
            )
        )
        fig.update_layout(
            xaxis_title="training variants",
            yaxis_title="variants",
        )
        _write_plot(fig, output_formats, base_path, "training-samples-importance")


def _make_predictions(batch_converter, dataset, model):
    predictions_mean = []
    predictions_std = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Predicting"):
            _, _, batch_tokens = batch_converter([("placeholder", dataset.variants[i])])
            prediction_mean, prediction_std = model(
                batch_tokens.to(model.train_embeddings.data.device), return_std=True
            )
            predictions_mean.append(prediction_mean.cpu().numpy()[0])
            predictions_std.append(prediction_std.cpu().numpy()[0])
    return np.asarray(predictions_mean), np.asarray(predictions_std)


def _should_plot(output_formats: List[str]) -> bool:
    return any(output_format.startswith("plot_") for output_format in output_formats)


def _write_plot(fig: go.Figure, output_formats: List[str], base_path: str, report_id: str):
    if "plot_html" in output_formats:
        fig.write_html("%s-%s-plot.html" % (base_path, report_id))
    if "plot_pdf" in output_formats:
        fig.write_image("%s-%s-plot.pdf" % (base_path, report_id))
    if "plot_json" in output_formats:
        fig.write_json("%s-%s-plot.json" % (base_path, report_id))
    if "plot_browser" in output_formats:
        fig.show()

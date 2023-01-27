import csv
import math
import sys
import time

import click
import fsspec
import numpy as np
import questionary
import sklearn.model_selection
import torch
from tqdm.auto import tqdm

from transmep import criterion_optimization, embed, get_device, reports
from transmep.data import (
    DatasetError,
    get_mutations_notation,
    load_dataset,
    load_wildtype,
    write_dataset,
)
from transmep.foundation_model import FOUNDATION_MODELS, load_foundation_model
from transmep.model import Model, load_model, save_model
from transmep.optimize import grid_search_kr_rbf_batched, grid_search_kr_rbf_fork
from transmep.questionary_validation import is_percentage, is_positive_int


@click.group()
def cli():
    pass


@cli.command(help="Split into training and test dataset")
def split():
    wildtype_path = questionary.text("Path to wild type sequence file").ask()
    variants_path = questionary.text("Path to original dataset").ask()
    variants_path_parts = variants_path.rsplit(".", 1)
    train_path = questionary.text(
        "Path to training dataset output", default="%s-train.%s" % tuple(variants_path_parts)
    ).ask()
    test_path = questionary.text(
        "Path to test dataset output", default="%s-test.%s" % tuple(variants_path_parts)
    ).ask()
    test_fraction = (
        float(
            questionary.text(
                "Size of the test dataset in percent", default="25", validate=is_percentage()
            ).ask()
        )
        / 100
    )

    dataset = load_dataset(wildtype_path, variants_path)
    train_idx, test_idx = sklearn.model_selection.train_test_split(
        np.arange(len(dataset)), test_size=test_fraction
    )
    train = dataset[train_idx]
    test = dataset[test_idx]

    write_dataset(train, train_path)
    write_dataset(test, test_path)


@cli.command(help="Download the ESM model in advance")
@click.option(
    "--foundation-model",
    help="ID of the foundation model to use.",
    default="esm2_t30_150M_UR50D",
    show_default=True,
    type=click.Choice(FOUNDATION_MODELS),
)
def download_esm(foundation_model):
    load_foundation_model(foundation_model)


@cli.command(help="Train a new model")
@click.option(
    "--model-path",
    prompt="Path to save the model to (model-path)",
    help="Path to save the model to",
    type=str,
)
@click.option(
    "--wildtype-path",
    prompt="Path to the wild type sequence file (wildtype-path)",
    help="Path to the wild type sequence file",
    type=str,
)
@click.option(
    "--variants-path",
    prompt="Path to the variants for training (variants-path)",
    help="Path to the variants for training",
    type=str,
)
@click.option(
    "--alpha-min",
    prompt="Minimum alpha hyperparameter (alpha-min)",
    help="Minimum alpha hyperparameter",
    default=1e-4,
    type=float,
)
@click.option(
    "--alpha-max",
    prompt="Maximum alpha hyperparameter (alpha-max)",
    help="Maximum alpha hyperparameter",
    default=1e5,
    type=float,
)
@click.option(
    "--alpha-steps",
    prompt="Number of alpha steps during grid search (alpha-steps)",
    help="Number of alpha steps during grid search",
    default=50,
    type=int,
)
@click.option(
    "--gamma-min",
    prompt="Minimum gamma hyperparameter (gamma-min)",
    help="Minimum gamma hyperparameter",
    default=1e-3,
    type=float,
)
@click.option(
    "--gamma-max",
    prompt="Maximum gamma hyperparameter (gamma-max)",
    help="Maximum gamma hyperparameter",
    default=1e6,
    type=float,
)
@click.option(
    "--gamma-steps",
    prompt="Number of gamma steps during grid search (gamma-steps)",
    help="Number of gamma steps during grid search",
    default=50,
    type=int,
)
@click.option(
    "--validation-iterations",
    prompt="Number of iterations for repeated holdout during validation (validation-iterations)",
    help="Number of iterations for repeated holdout during validation",
    default=1000,
    type=int,
)
@click.option(
    "--batch-size",
    prompt="Number of validation iterations to process in one batch (batch-size)",
    help="Number of validation iterations to process in one batch",
    default=100,
    type=int,
)
@click.option(
    "--block-size",
    prompt="Block size to process in one batch for distance matrix calculation (block-size)",
    help="Block size to process in one batch for distance matrix calculation (block-size)",
    default=100,
    type=int,
)
@click.option(
    "--holdout-fraction",
    prompt="Fraction of samples to use for validation during repeated holdout (holdout-fraction)",
    help="Fraction of samples to use for validation during repeated holdout",
    default=0.1,
    type=float,
)
@click.option(
    "--grid-search-output",
    help="Path to save grid search output to, pass empty string for no output (grid-search-output)",
    prompt="Path to save grid search output to, pass empty string for no output",
    type=str,
)
@click.option(
    "--foundation-model",
    help="ID of the foundation model to use.",
    default="esm2_t30_150M_UR50D",
    show_default=True,
    type=click.Choice(FOUNDATION_MODELS),
)
def train(
    model_path: str,
    wildtype_path: str,
    variants_path: str,
    alpha_min: float,
    alpha_max: float,
    alpha_steps: int,
    gamma_min: float,
    gamma_max: float,
    gamma_steps: int,
    validation_iterations: int,
    batch_size: int,
    block_size: int,
    holdout_fraction: float,
    grid_search_output: str,
    foundation_model: str,
):
    click.echo(click.style("Welcome to TransMEP!", fg="green"))

    click.echo("Loading dataset")
    dataset = load_dataset(wildtype_path, variants_path)
    y = torch.tensor(dataset.y)

    click.echo("Loading protein language model (%s)" % foundation_model)
    transformer, alphabet = load_foundation_model(foundation_model)

    click.echo("Embedding training variants")
    embeddings = embed(dataset, transformer, alphabet.get_batch_converter())

    click.echo("Performing grid search for hyper parameters")
    training_size = int(len(dataset) * (1 - holdout_fraction))
    alphas = torch.logspace(math.log10(alpha_min), math.log10(alpha_max), alpha_steps, base=10)
    gammas = torch.logspace(math.log10(gamma_min), math.log10(gamma_max), gamma_steps, base=10)
    prior_time = time.time()
    if torch.cuda.is_available():
        best_alpha, best_gamma, scores = grid_search_kr_rbf_batched(
            embeddings,
            y,
            alphas,
            gammas,
            validation_iterations,
            training_size,
            batch_size,
            block_size,
            get_device(),
        )
    else:
        best_alpha, best_gamma, scores = grid_search_kr_rbf_fork(
            embeddings, y, alphas, gammas, validation_iterations, training_size
        )
    grid_search_time = time.time() - prior_time
    click.echo("Grid search wall time: %.4fs" % grid_search_time)
    if grid_search_output != "":
        with fsspec.open(grid_search_output, "wb") as fd:
            np.savez(
                fd,
                scores=scores.cpu().numpy(),
                alphas=alphas.cpu().numpy(),
                gammas=gammas.cpu().numpy(),
            )

    click.echo("Fitting final model")
    with torch.no_grad():
        model = Model(foundation_model, transformer=transformer)
        model.fit(best_alpha, best_gamma, embeddings, y, device=get_device(), block_size=block_size)
    save_model(model, model_path)

    click.echo(
        click.style(
            "Final model trained & saved!",
            fg="green",
        )
    )


@cli.command(help="Predict new variants")
@click.option(
    "--model-path",
    prompt="Path to load the model from",
    help="Path to load the model from",
    type=str,
)
@click.option(
    "--wildtype-path",
    prompt="Path to the wild type sequence file",
    help="Path to the wild type sequence file",
    type=str,
)
@click.option(
    "--variants-path",
    prompt="Path to the variants for inference",
    help="Path to the variants for inference",
    type=str,
)
@click.option(
    "--output-path",
    prompt="Path to save the output to",
    help="Path to save the output to",
    type=str,
)
@click.option(
    "--batch-size",
    help="Batch size for inference",
    type=int,
    default=100,
)
def predict(
    model_path: str, wildtype_path: str, variants_path: str, output_path: str, batch_size: int
):
    model, alphabet = load_model(model_path)
    device = get_device()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    dataset = load_dataset(wildtype_path, variants_path)

    with torch.no_grad():
        with fsspec.open(output_path, "w") as fd:
            writer = csv.DictWriter(fd, fieldnames=["mutations", "prediction", "prediction_std"])
            writer.writeheader()
            for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Predicting"):
                batch_end = min(len(dataset), batch_start + batch_size)
                variants = dataset.variants[batch_start:batch_end]
                mutants = [
                    get_mutations_notation(dataset.wildtype, variant) for variant in variants
                ]
                _, _, batch_tokens = batch_converter(
                    [("placeholder", variant) for variant in variants]
                )
                prediction_mean, prediction_std = model(batch_tokens.to(device), return_std=True)
                for mutant, mean, std in zip(mutants, prediction_mean, prediction_std):
                    writer.writerow(
                        dict(
                            mutations=mutant,
                            prediction=float(mean),
                            prediction_std=float(std),
                        )
                    )


@cli.command(help="Create a detailed report")
def report():
    report_ids = questionary.checkbox(
        "Which reports do you want to create?",
        [
            questionary.Choice("R² score (r2)", value="r2"),
            questionary.Choice(
                "Mutation attribution, requires backpropagation through ESM (mutation_attribution)",
                value="mutation_attribution",
            ),
            questionary.Choice("Grid search report (grid_search)", value="grid_search"),
            questionary.Choice(
                "Training samples importance (training_samples_importance)",
                value="training_samples_importance",
            ),
        ],
    ).ask()
    output_formats = questionary.checkbox(
        "What output formats do you want? Output formats not supported by a report will be skipped quietly.",
        [
            questionary.Choice("JSON", "json"),
            questionary.Choice("Plot (HTML)", "plot_html"),
            questionary.Choice("Plot (JSON)", "plot_json"),
            questionary.Choice("Plot (show in browser)", "plot_browser"),
            questionary.Choice("Console", "console"),
        ],
    ).ask()
    params = {}
    if (
        "r2" in report_ids
        or "mutation_attribution" in report_ids
        or "training_samples_importance" in report_ids
    ):
        params["model_path"] = questionary.text("Model path").ask()
        params["wildtype_path"] = questionary.text("Path to wild type sequence file").ask()
        params["variants_path"] = questionary.text("Path to variants file").ask()
    if "mutation_attribution" in report_ids:
        params["attribution_steps"] = int(
            questionary.text(
                "Number of steps to use for integral approximation in integrated gradients",
                validate=is_positive_int(),
                default="100",
            ).ask()
        )
    if "grid_search" in report_ids:
        params["grid_path"] = questionary.text("Path to the grid search output").ask()
    if "training_samples_importance" in report_ids:
        params["train_variants_path"] = questionary.text("Path to training variants file").ask()
    if len(set(output_formats).difference(["console", "plot_browser"])) != 0:
        params["base_path"] = questionary.text("Report file prefix").ask()
    else:
        params["base_path"] = ""
    report_map = {
        "r2": reports.r2,
        "mutation_attribution": reports.mutation_attribution,
        "grid_search": reports.grid_search,
        "training_samples_importance": reports.training_samples_importance,
    }
    for report_id in report_ids:
        click.echo("Creating report %s" % report_id)
        report_map[report_id](output_formats=output_formats, **params)
    click.echo(click.style("All reports created!", fg="green"))


@cli.command(help="Optimize the UCB acquisition criterion")
@click.option(
    "--model-path",
    prompt="Path to load the model from",
    help="Path to load the model from",
    type=str,
)
@click.option(
    "--wildtype-path",
    prompt="Path to the wild type sequence file",
    help="Path to the wild type sequence file",
    type=str,
)
@click.option(
    "--batch-size",
    help="Batch size for model inference",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--kappa",
    help="Kappa value for UCB. Higher kappa values lead to more exploration.",
    prompt="Kappa value for UCB. Higher kappa values lead to more exploration",
    type=float,
)
@click.option(
    "--population-size",
    help="Size of population per restart.",
    prompt="Size of population per restart",
    type=int,
)
@click.option(
    "--restarts",
    help="How many initializations to try for genetic optimization.",
    prompt="How many initializations to try for genetic optimization",
    type=int,
)
@click.option(
    "--num-mutations",
    help="Maximum number of mutations to allow.",
    prompt="Maximum number of mutations to allow.",
    type=int,
)
@click.option(
    "--sites",
    help="Comma separated list of positions that should be mutated. Set to 'all' to allow mutations on all positions.",
    prompt="Comma separated list of positions that should be mutated. Set to 'all' to allow mutations on all positions",
    type=str,
)
@click.option(
    "--mutation-probability",
    help="Probability of each position to mutate. Defaults to the inverse of the sequence length.",
    default=-1.0,
)
@click.option(
    "--crossover-probability",
    help="Probability to exchange information at each position.",
    default=0.5,
    show_default=True,
)
@click.option(
    "--min-diversity",
    help="Minimum fraction of distinct mutants in the population before stopping optimization.",
    default=0.1,
    show_default=True,
)
@click.option(
    "--max-generations",
    help="Maximum number of generations per restart.",
    default=100,
    show_default=True,
)
def ucb(
    model_path: str,
    wildtype_path: str,
    batch_size: int,
    kappa: float,
    population_size: int,
    restarts: int,
    num_mutations: int,
    sites: str,
    mutation_probability: float,
    crossover_probability: float,
    min_diversity: float,
    max_generations: int,
):
    device = get_device()
    wildtype = load_wildtype(wildtype_path)
    if sites == "all":
        sites = np.arange(len(wildtype))
    else:
        sites = np.asarray([int(p) - 1 for p in sites.split(",")])
    if mutation_probability == -1.0:
        mutation_probability = None
    model, alphabet = load_model(model_path)
    model.to(device)
    evaluator = criterion_optimization.CachingEvaluator(
        alphabet,
        model,
        wildtype,
        criterion_optimization.criterion_ucb,
        batch_size=batch_size,
        kappa=kappa,
        device=device,
    )
    best_sequences, best_values = criterion_optimization.genetic_optimization_restarts(
        wildtype,
        evaluator,
        population_size,
        restarts,
        num_mutations,
        sites=sites,
        mutation_probability=mutation_probability,
        crossover_probability=crossover_probability,
        min_diversity=min_diversity,
        max_generations=max_generations,
    )
    order = np.argsort(best_values)[::-1]
    for i, index in enumerate(order):
        print(
            "Rank %d with UCB = %.4f\n%s"
            % (i, best_values[index], get_mutations_notation(wildtype, best_sequences[index]))
        )


def main():
    try:
        cli()
    except DatasetError as e:
        click.echo(click.style(e.message, fg="red"))
        sys.exit(1)

# TransMEP - Transfer learning for Mutant Effect Prediction

TransMep is a tool for using transfer learning embeddings from protein language models to train variant prediction models from existing mutagenesis data.
It is focused on speed and simplicity of use.
You just input your dataset and obtain a prediction model, accompanied by detailed reports on performance, hyperparameter optimization, training samples importance and even an attribution to individual mutations.


## Publication & Citation
[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.01.12.575432v1)
```
@article{Hoffbauer2024,
  title = {TransMEP: Transfer learning on large protein language models to predict mutation effects of proteins from a small known dataset},
  url = {http://dx.doi.org/10.1101/2024.01.12.575432},
  DOI = {10.1101/2024.01.12.575432},
  publisher = {Cold Spring Harbor Laboratory},
  author = {Hoffbauer,  Tilman and Strodel,  Birgit},
  year = {2024},
  month = jan 
}
```


## Installation

**Tip:** You want to skip the hassle of installation or do not have a NVIDIA GPU? Just use the [Google Colab notebook](https://colab.research.google.com/drive/1A3jV5-M264GdjQ8kCybkXTxUJV8CRDSJ?usp=sharing). You get limited access to free GPUs there.

TransMEP's main dependency is PyTorch, which you should install first ([Guide](https://pytorch.org/get-started/locally/)).
While TransMEP does not require an NVIDIA GPU, it is significantly faster with it.
Usually, setting up Torch with GPU acceleration via CUDA is worth the hassle.
After that, just install TransMEP on top:

```bash
pip install transmep
```

## General workflow
For a wet-lab study, we recommend the following workflow:

1. Sample an initial set of mutants that seed future optimization. For example, one could select the wild type and 9 random mutants with a single mutation. Evaluate these mutants in the lab and collect their target values.
2. Train an initial model.
3. View the available reports. Note that computing R² values is not applicable in this setting.
4. Determine the next variant via the UCB criterion.
5. Test it in the lab, measure its target value and go back to step 2 until a sufficiently high target value is achieved.

Each step corresponds to one of the commands explained below.

*If you plan on using TransMEP, please get in touch with us.*

## Command reference

### Training a model
```
transmep train
```

You will be prompted for all relevant parameters if you do not specify them on the command line (for help see `transmep --help` and `transmep train --help`).

- `model-path` Your model will be saved as a Torch state dict to this location. All paths can be both local or remote, for more details see [fsspec](https://filesystem-spec.readthedocs.io/en/latest/index.html).
- `wildtype-path` Path to the wildtype sequence. Here, TransMEP expects a text file with the sequence in the amino acid single letter code, including `X` for rare amino acids.
- `variants-path` Path to the known variants that should be used for training (Have a look at `transmep split` for splitting datasets into training and test subsets). Here, TransMEP expects a CSV with at least two columns, named `mutations` and `y`. While `y` is just the target value, `mutations` should contain the list of mutations of each mutant in the format `A123B+C234D` (the wildtype is an empty string).
- `alpha-min`, `alpha-max` These parameters describe the $\alpha$ values that are tried during hyperparameter optimization. The $\alpha$ value controls the regularization of the model - the higher the value, the worse the fit will be, but the more unlikely is overfitting. Note that the training process can become numerically unstable for very low $\alpha$ values.
- `gamma-min`, `gamma-max` This parameter is a scale parameter, i.e. how correlated two samples are if they are close in the embedding space.
- `alpha-steps`, `gamma-steps` The resolution of the grid search. Note that the runtime grows linear with the product of `alpha-steps` and `gamma-steps`.
- `validation-iterations` Number of iterations for repeated holdout during validation. Higher values give estimates for the generalization error with lower variance, but increase runtime.
- `batch-size` Number of validation iterations to process in one batch. Lower this value if you get CUDA out of memory errors, and increase it if GPU utilization is low.
- `holdout-fraction` Fraction of samples for holdout during validation. Higher values also decrease the variance of the generalization error estimate, but they increase the bias of the estimate. Usually, you do not want to change this value.
- `grid-search-output` Path to save the output of the grid search report to. If you pass an empty string, nothing will be saved. This file can be used for the reports later on.


### Predicting new variants
```
transmep predict
```

This command can be used for predicting the target value for new variants.
The variants file is again expected to be a CSV, but this time only the `mutations` column is required.
The output contains the columns `prediction` (estimate for the target value) and `prediction_std` (estimate for the standard deviation of the prediction).
The latter is calculated under the assumption that the RBF kernel used in TransMEP is a correct fit for fitness landscape, so this value should be handled with care and only used for comparing predictions.

### Determining the next variant
```
transmep ucb
```

With this command, you can start the search for a variant with a high UCB value should be evaluated next.
It starts the genetic algorithm using the following parameters:

- `model-path` Path to the model from `transmep train`.
- `wildtype-path` Path to the wild type sequence.
- `batch-size` Size of the batches used for inference on the model to calculate the missing UCB values.
- `kappa` Kappa value in the UCB criterion.
- `population-size` Size of the population in the genetic algorithm.
- `restarts` How many repetitions of the genetic algorithm should be performed.
- `num_mutations` Maximum number of mutations per mutant.
- `sites` Comma separated list of positions that should be mutated. Set to `all` to allow mutations on all positions.
- `mutation-probability` Hyperparameter of the genetic algorithm. If not set, this is set to `1 / number of sites`.
- `crossover-probability` Hyperparameter of the genetic algorithm. If not set, this defaults to 0.5.
- `min-diversity` Minimum fraction of distinct mutants in the population before stopping optimization. This defaults to 0.1.
- `max-generations` Maximum number of generations per genetic algorithm repetition. This defaults to 100.

Depending on your available computing time, we recommend to try some variations of the `mutation-probability`, `crossover-probability`, `min-diversity` and `max-generations` parameters to find the candidate with the highest UCB value.


### Reports
```
transmep reports
```

For `variants-path`, one should now pass a set of new variants, e.g. the test dataset.
The supported reports are:

- `r2` Estimate the coefficient of determination of this model, i.e. the fraction of variance in the dataset explained by the model. This also reports a confidence interval which is based on bootstrapping and the standard deviation estimated by the model.
- `mutation_attribution` This report estimates the effect of every single mutation of a mutant on the total value. The y-axis contains the mutations while the x-axis contains the variants. Each column sums up to the predicted target value of the variant.
- `grid_search` Here, one can observe the estimated generalization error during grid search for various hyperparameter valuations ($\alpha$ and $\gamma$). If the optimum is close to the border, rerun the training process with larger parameter ranges. The colors are on a logarithmic scale.
- `training_samples_importance` This report calculates the importance of each training sample for the prediction of a variant. Each row sums up to 1.

## Usage Example
This example is based on a dataset from [[Wu et al. 2019]](https://doi.org/10.1073/pnas.1901979116), which was also used in my [bachelor thesis as C75](https://github.com/Turakar/mutation-prediction).

Let's first train a model on the provided example data.
```
$ transmep train
Path to save the model to (model-path): https://mutation-prediction.thoffbauer.de/transmep/c-wt.txt
Path to the wild type sequence file (wildtype-path): ^CAborted!
(transmep-py3.10) [t@tpc transmep-publication]$ transmep train
Path to save the model to (model-path): mymodel.pt
Path to the wild type sequence file (wildtype-path): https://mutation-prediction.thoffbauer.de/transmep/c-wt.txt
Path to the variants for training (variants-path): https://mutation-prediction.thoffbauer.de/transmep/c-variants-train.csv
Minimum alpha hyperparameter (alpha-min) [0.0001]: 
Maximum alpha hyperparameter (alpha-max) [100000.0]: 
Number of alpha steps during grid search (alpha-steps) [50]: 
Minimum gamma hyperparameter (gamma-min) [0.001]: 
Maximum gamma hyperparameter (gamma-max) [1000000.0]: 
Number of gamma steps during grid search (gamma-steps) [50]: 
Number of iterations for repeated holdout during validation (validation-iterations) [1000]: 
Number of validation iterations to process in one batch (batch-size) [100]: 
Block size to process in one batch for distance matrix calculation (block-size) [100]: 
Fraction of samples to use for validation during repeated holdout (holdout-fraction) [0.1]: 
Path to save grid search output to, pass empty string for no output: mymodel-grid.npz
Welcome to TransMEP!
Loading dataset
Loading protein language model (esm2_t30_150M_UR50D)
Embedding training variants
Embedding: 100%|██████████| 424/424 [00:10<00:00, 40.72it/s]
Performing grid search for hyper parameters
HPO: 100%|██████████| 10/10 [02:21<00:00, 14.12s/it]
Grid search wall time: 141.4151s
Fitting final model
Final model trained & saved!
```

We will also check the grid search report.

```
$ transmep report
? Which reports do you want to create? [Grid search report (grid_search)]
? What output formats do you want? Output formats not supported by a report will be skipped quietly. done (2 selecti
ons)
? Path to the grid search output mymodel-grid.npz
? Report file prefix mymodel
Creating report grid_search
All reports created!
```

You can find the report [here](https://mutation-prediction.thoffbauer.de/transmep/c-grid-search-plot.html).
As we are satisfied with the result, we move on to finding promising new candidates:

```
$ transmep ucb
Path to load the model from: mymodel.pt
Path to the wild type sequence file: https://mutation-prediction.thoffbauer.de/transmep/c-wt.txt
Kappa value for UCB. Higher kappa values lead to more exploration: 3
Size of population per restart: 100
How many initializations to try for genetic optimization: 10
Maximum number of mutations to allow.: 5
Comma separated list of positions that should be mutated. Set to 'all' to allow mutations on all positions: 32,46,49,51,53,56,97
Criterion optimization: 100%|██████████| 10/10 [01:13<00:00,  7.39s/it]
Rank 0 with UCB = 1.1940
Y32G+F46S+I53D+L56K+V97N
Rank 1 with UCB = 1.1917
Y32G+F46S+I53G+L56K+V97E
Rank 2 with UCB = 1.1215
Y32A+F46S+I53T+L56A+V97E
Rank 3 with UCB = 1.1201
Y32S+F46A+I53D+L56A+V97R
Rank 4 with UCB = 1.0993
Y32E+I53D+L56V+V97R
Rank 5 with UCB = 1.0981
Y32I+I53D+L56S+V97S
Rank 6 with UCB = 1.0935
Y32T+I53D+V97R
Rank 7 with UCB = 1.0876
Y32S+I53E+V97R
Rank 8 with UCB = 1.0827
Y32P+F46Q+I53S+L56G+V97E
Rank 9 with UCB = 1.0812
Y32G+I53P+L56V+V97E
```


## Contributing
This project uses [Poetry](https://python-poetry.org/) for dependency management, so please install this first.
Then, you can install all dependencies using this command:

```
poetry install
```

And open a shell inside the virtual environment using:

```
poetry shell
```

If you make some changes, please create a PR to merge them into this repository. Also, please ensure that your code is well formatted by running `black .` and `isort .`. Tests can be executed using `pytest`.

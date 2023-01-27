import copy
import functools
from collections import OrderedDict
from typing import List, Protocol, Tuple

import esm
import numpy as np
import torch
from deap import tools
from deap.base import Fitness as FitnessBase
from tqdm.auto import tqdm

from transmep.model import Model

amino_acid_alphabet = list("RHKDESTNQCUGPAVILMFYW")


class Criterion(Protocol):
    def __call__(self, mean: torch.Tensor, std: torch.Tensor, **kwargs) -> torch.Tensor:
        ...


def criterion_ucb(mean: torch.Tensor, std: torch.Tensor, **kwargs) -> torch.Tensor:
    return mean + kwargs["kappa"] * std


class FitnessMaxSingle(FitnessBase):
    weights = (1.0,)


class Individual(list):
    def __init__(self, x: List[Tuple[int, int]]):
        super(Individual, self).__init__(x)
        self.fitness = FitnessMaxSingle()

    def to_sequence(self, wildtype: str) -> str:
        variant = list(wildtype)
        for p, a in self:
            variant[p] = amino_acid_alphabet[a]
        return "".join(variant)

    def mutated_positions(self) -> set:
        return {p for p, _ in self}


class CachingEvaluator:
    def __init__(
        self,
        alphabet: esm.Alphabet,
        model: Model,
        wildtype: str,
        criterion: Criterion,
        batch_size: int = 100,
        device: torch.device = torch.device("cuda:0"),
        budget: int = None,
        **kwargs
    ):
        self.batch_converter = alphabet.get_batch_converter()
        model.to(device)
        self.model = model
        self.wildtype = wildtype
        self.criterion = functools.partial(criterion, **kwargs)
        self.batch_size = batch_size
        self.device = device
        self.budget = budget
        self.cache = dict()

    @torch.no_grad()
    def evaluate(self, population: List[Individual]) -> None:
        # this function evaluates a batch and saves its results
        def evaluate_batch(batch):
            _, _, batch_tokens = self.batch_converter(
                [("foo", individual.to_sequence(self.wildtype)) for individual in batch]
            )
            mean, std = self.model(batch_tokens.to(self.device), return_std=True)
            values = self.criterion(mean, std).cpu()
            for individual, value in zip(batch, values):
                value = float(value)
                individual.fitness.values = (value,)
                self.cache[frozenset(individual)] = value

        # iterate over all individuals and evaluate missing values in batches
        todo = [individual for individual in population if not individual.fitness.valid]
        current_batch = []
        while len(todo) > 0:

            # next candidate
            individual = todo.pop()

            # check cache
            cached_entry = self.cache.get(frozenset(individual))
            if cached_entry:
                individual.fitness.values = (cached_entry,)
                continue

            # check budget
            if self.budget is not None:
                if self.budget <= 0:
                    raise BudgetError("Available evaluator budget is used up.")
                else:
                    self.budget -= 1

            # add to current batch and evaluate it if full
            current_batch.append(individual)
            if len(current_batch) == self.batch_size:
                evaluate_batch(current_batch)
                current_batch = []

        # evaluate remaining batch
        if len(current_batch) > 0:
            evaluate_batch(current_batch)


class BudgetError(Exception):
    pass


class ConstrainedGeneticOperations:
    def __init__(
        self,
        wildtype: str,
        crossover_probability: float,
        mutation_probability: float,
        num_mutations: int,
        sites: np.ndarray,
        rng: np.random.Generator,
    ):
        self.wildtype = wildtype
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.num_mutations = num_mutations
        self.sites = sites
        self.rng = rng

    def init_population(self, size: int) -> List[Individual]:
        wildtype_mutations = [(p, amino_acid_alphabet.index(self.wildtype[p])) for p in self.sites]
        wildtype = Individual(
            [
                wildtype_mutations[i]
                for i in self.rng.choice(len(self.sites), size=self.num_mutations, replace=False)
            ]
        )
        population = [wildtype]
        for _ in range(size - 1):
            variant = Individual(
                [
                    wildtype_mutations[i]
                    for i in self.rng.choice(
                        len(self.sites), size=self.num_mutations, replace=False
                    )
                ]
            )
            variant[self.rng.integers(0, self.num_mutations)] = self._random_mutation()
            population.append(variant)
        return population

    def mutate(self, individual: Individual) -> None:
        for i in range(self.num_mutations):
            if self.rng.random() < self.mutation_probability:
                p, a = self._random_mutation()
                if p not in individual.mutated_positions().difference({individual[i][0]}):
                    individual[i] = (p, a)

    def mate(self, individual1: Individual, individual2: Individual) -> None:
        for i in range(self.num_mutations):
            if self.rng.random() < self.crossover_probability:
                p1 = individual1[i][0]
                p2 = individual2[i][0]
                if p1 == p2 or (
                    p2 not in individual1.mutated_positions().difference({p1})
                    and p1 not in individual2.mutated_positions().difference({p2})
                ):
                    individual1[i], individual2[i] = individual2[i], individual1[i]

    def _random_mutation(self) -> Tuple[int, int]:
        return self.rng.choice(self.sites), self.rng.integers(0, len(amino_acid_alphabet))


def genetic_optimization_restarts(
    wildtype: str,
    evaluator: CachingEvaluator,
    population_size: int,
    restarts: int,
    num_mutations: int,
    sites: np.ndarray = None,
    mutation_probability: float = None,
    crossover_probability: float = 0.5,
    rng: np.random.Generator = None,
    show_progress_bar: bool = True,
    min_diversity: float = 0.01,
    max_generations: int = 100,
) -> Tuple[List[str], List[float]]:

    # Iterate over the restarts and collect best variant
    best_sequences = []
    best_values = []
    iterator = range(restarts)
    if show_progress_bar:
        iterator = tqdm(iterator, desc="Criterion optimization")
    for _ in iterator:
        sequence, value = genetic_optimization(
            wildtype,
            evaluator,
            population_size,
            num_mutations,
            sites=sites,
            mutation_probability=mutation_probability,
            crossover_probability=crossover_probability,
            rng=rng,
            min_diversity=min_diversity,
            max_generations=max_generations,
            show_progress_bar=False,
        )
        best_sequences.append(sequence)
        best_values.append(value)

    return best_sequences, best_values


def genetic_optimization(
    wildtype: str,
    evaluator: CachingEvaluator,
    population_size: int,
    num_mutations: int,
    sites: np.ndarray = None,
    mutation_probability: float = None,
    crossover_probability: float = 0.5,
    rng: np.random.Generator = None,
    min_diversity: float = None,
    max_generations: int = 100,
    show_progress_bar: bool = True,
):
    # Argument parsing
    if sites is None:
        sites = np.arange(len(wildtype))
    if rng is None:
        rng = np.random.default_rng()
    if mutation_probability is None:
        mutation_probability = 1 / len(sites)
    if min_diversity is None:
        min_diversity = 2 / population_size
    else:
        min_diversity = max(2 / population_size, min_diversity)
    operations = ConstrainedGeneticOperations(
        wildtype,
        crossover_probability,
        mutation_probability,
        num_mutations,
        sites,
        rng,
    )

    # Init population
    population = operations.init_population(population_size)
    evaluator.evaluate(population)

    # Iterate over generations
    iterator = range(max_generations)
    if show_progress_bar:
        # hide total value as it is uninformative
        iterator = tqdm((i for i in iterator), desc="Generations")
    for _ in iterator:

        # Select offspring
        offspring = [
            copy.deepcopy(individual)
            for individual in tools.selTournament(population, len(population), 2)
        ]

        # Vary population
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            operations.mate(ind1, ind2)
            operations.mutate(ind1)
            operations.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate new fitness values
        try:
            evaluator.evaluate(offspring)
        except BudgetError:
            if show_progress_bar:
                iterator.close()
            break

        # Select new population
        population = tools.selBest(population + offspring, population_size)

        # Check diversity
        fraction_distinct = len({tuple(individual) for individual in population}) / len(population)
        if show_progress_bar:
            best_fitness = max(individual.fitness.values[0] for individual in population)
            iterator.set_postfix(
                OrderedDict((("fitness", best_fitness), ("distinct", fraction_distinct)))
            )
        if fraction_distinct < min_diversity:
            if show_progress_bar:
                iterator.close()
            break

    # Find maximum
    best = max(population, key=lambda ind: ind.fitness.values[0])
    return best.to_sequence(wildtype), best.fitness.values[0]

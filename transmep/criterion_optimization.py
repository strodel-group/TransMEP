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

# This list does not contain the unknown amino acid X.
amino_acid_alphabet = list("RHKDESTNQCUGPAVILMFYW")

# This files implements the UCB criterion optimization via genetic algorithms.


class Criterion(Protocol):
    """
    Function signature for a criterion.
    """

    def __call__(self, mean: torch.Tensor, std: torch.Tensor, **kwargs) -> torch.Tensor:
        ...


def criterion_ucb(mean: torch.Tensor, std: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    The UCB criterion.

    :param mean: Predicted means.
    :param std: Predicted standard deviations.
    :param kwargs: Must contain a value for kappa.
    :return: The UCB criterion values.
    """
    return mean + kwargs["kappa"] * std


class FitnessMaxSingle(FitnessBase):
    """
    This class defines the fitness objective for DEAP.
    """

    weights = (1.0,)


class Individual(list):
    """
    This class defines each individual in the population of the genetic algorithm for DEAP.
    """

    def __init__(self, x: List[Tuple[int, int]]):
        """
        Create a new individual.

        :param x: The list of mutations (position, acid)
        """
        super(Individual, self).__init__(x)
        self.fitness = FitnessMaxSingle()

    def to_sequence(self, wildtype: str) -> str:
        """
        Convert this individual to a sequence string.

        :param wildtype: The wildtype sequence.
        :return: The individual's sequence.
        """
        variant = list(wildtype)
        for p, a in self:
            variant[p] = amino_acid_alphabet[a]
        return "".join(variant)

    @property
    def mutated_positions(self) -> set:
        """
        All mutated positions.
        """
        return {p for p, _ in self}


class CachingEvaluator:
    """
    Utility class for evaluating UCB values of individuals.
    Each object of this class caches all evaluations and takes care of batching.
    """

    def __init__(
        self,
        alphabet: esm.Alphabet,
        model: Model,
        wildtype: str,
        criterion: Criterion,
        batch_size: int = 100,
        device: torch.device = torch.device("cpu"),
        budget: int = None,
        **kwargs
    ):
        """
        Create a new CachingEvaluator.

        :param alphabet: Alphabet of the model.
        :param model: TransMEP model.
        :param wildtype: Wildtype sequence.
        :param criterion: Criterion, e.g. UCB.
        :param batch_size: Batch size for inference.
        :param device: Device to perform inference on.
        :param budget: Maximum number of evaluations to perform before raising a BudgetError.
            Set to None to allow arbitrary many evaluations.
        :param kwargs: kwargs for the criterion, e.g. kappa for UCB.
        """
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
        """
        Ensure that all individuals in this population have a fitness value.

        :param population: The population.
        :return: None. Fitness values are saved in the individuals.
        """
        # Utility function that evaluates a given batch of individuals, saves their fitness value
        # and adds them to the cache.
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

        # Iterate over all individuals and evaluate missing values in batches
        todo = [individual for individual in population if not individual.fitness.valid]
        current_batch = []
        while len(todo) > 0:

            # Next candidate
            individual = todo.pop()

            # Check cache
            cached_entry = self.cache.get(frozenset(individual))
            if cached_entry:
                individual.fitness.values = (cached_entry,)
                continue

            # Check budget
            if self.budget is not None:
                if self.budget <= 0:
                    raise BudgetError("Available evaluator budget is used up.")
                else:
                    self.budget -= 1

            # Add to current batch and evaluate it if full
            current_batch.append(individual)
            if len(current_batch) == self.batch_size:
                evaluate_batch(current_batch)
                current_batch = []

        # Evaluate last batch
        if len(current_batch) > 0:
            evaluate_batch(current_batch)


class BudgetError(Exception):
    """
    Thrown by CachingEvaluator if its budget is used up.
    """

    pass


class ConstrainedGeneticOperations:
    """
    This class implements the allowed operations on the individuals.
    These operations are constrained because we may not introduce more than two mutations for one position.
    """

    def __init__(
        self,
        wildtype: str,
        crossover_probability: float,
        mutation_probability: float,
        num_mutations: int,
        sites: np.ndarray,
        rng: np.random.Generator,
    ):
        """
        Create a new ConstrainedGeneticOperations object.

        :param wildtype: Wildtype sequence.
        :param crossover_probability: Probability for crossover at each position.
        :param mutation_probability: Probability for mutation at each position.
        :param num_mutations: Maximum number of mutations (individual's length).
        :param sites: Sites allowed for mutation.
        :param rng: A NumPy RNG.
        """
        self.wildtype = wildtype
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.num_mutations = num_mutations
        self.sites = sites
        self.rng = rng

    def init_population(self, size: int) -> List[Individual]:
        """
        Creates an initial population with the wild type and individuals with a single mutation.

        :param size: Size of the initial population.
        :return: Initial population.
        """
        # Convert wildtype sequence to list of "mutations"
        wildtype_mutations = [(p, amino_acid_alphabet.index(self.wildtype[p])) for p in self.sites]

        # The wildtype specifies num_mutations of its amino acids explicitly.
        # These are randomly selected. The remaining positions are assumed to be the wild type implicitly.
        wildtype = Individual(
            [
                wildtype_mutations[i]
                for i in self.rng.choice(len(self.sites), size=self.num_mutations, replace=False)
            ]
        )

        # Create population
        population = [wildtype]
        for _ in range(size - 1):
            # Each individual specifies num_mutations amino acids explicitly.
            # Of these, one is mutated, and the rest is equal to the wild type.
            # The other positions are assumed to be the wild type implicitly.
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
        """
        Apply random mutations to this individual.
        If a newly introduced mutation would collide with an existing mutation, skip it.

        :param individual: Individual to mutate.
        :return: None. Mutations are performed in-place.
        """
        for i in range(self.num_mutations):
            if self.rng.random() < self.mutation_probability:
                p, a = self._random_mutation()
                if p not in individual.mutated_positions.difference({individual[i][0]}):
                    individual[i] = (p, a)

    def mate(self, individual1: Individual, individual2: Individual) -> None:
        """
        Apply crossover between two individuals.
        If a mutation introduced by an exchange would collide with an existing mutation, skip it.

        :param individual1: Individual A.
        :param individual2: Individual B.
        :return: None. Crossover is performed in place.
        """
        for i in range(self.num_mutations):
            if self.rng.random() < self.crossover_probability:
                p1 = individual1[i][0]
                p2 = individual2[i][0]
                if p1 == p2 or (
                    p2 not in individual1.mutated_positions.difference({p1})
                    and p1 not in individual2.mutated_positions.difference({p2})
                ):
                    individual1[i], individual2[i] = individual2[i], individual1[i]

    def _random_mutation(self) -> Tuple[int, int]:
        """
        Utility function for creating a random mutation.

        :return: (position, acid)
        """
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
    """
    Performs genetic optimization of the criterion with multiple restarts.

    :param wildtype: Wild type sequence.
    :param evaluator: CachingEvaluator to use. Allows re-use between multiple optimization runs.
    :param population_size: Size of the population.
    :param restarts: Number of repetitions to execute.
    :param num_mutations: Maximum number of mutations.
    :param sites: Positions allowed to mutate.
    :param mutation_probability: Per-site mutation probability during genetic optimization.
        Set to 1 / len(sites) if None.
    :param crossover_probability: Per-mutation crossover probability during genetic optimization.
    :param rng: NumPy RNG. Created if not provided.
    :param show_progress_bar: Whether to show a progress bar.
    :param min_diversity: Minimum fraction of different mutants in the population before early stopping.
    :param max_generations: Maximum number of generations per genetic optimization.
    :return: List of the best variants and their UCB values.
    """

    # Prepare iterator with a progress bar if enabled
    iterator = range(restarts)
    if show_progress_bar:
        iterator = tqdm(iterator, desc="Criterion optimization")

    # Iterate over the restarts and collect best variant
    best_sequences = []
    best_values = []
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
) -> Tuple[str, float]:
    """
    Performs a single run of genetic optimization of the criterion.

    :param wildtype: Wild type sequence.
    :param evaluator: CachingEvaluator to use. Allows re-use between multiple optimization runs.
    :param population_size: Size of the population.
    :param num_mutations: Maximum number of mutations.
    :param sites: Positions allowed to mutate.
    :param mutation_probability: Per-site mutation probability during genetic optimization.
        Set to 1 / len(sites) if None.
    :param crossover_probability: Per-mutation crossover probability during genetic optimization.
    :param rng: NumPy RNG. Created if not provided.
    :param show_progress_bar: Whether to show a progress bar.
    :param min_diversity: Minimum fraction of different mutants in the population before early stopping.
    :param max_generations: Maximum number of generations per genetic optimization.
    :return: Best variant and its UCB value.
    """
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

    # Prepare genetic operations toolbox
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
        # Show progress bar if enabled. Hide total value as it is uninformative
        iterator = tqdm((i for i in iterator), desc="Generations")
    for _ in iterator:

        # Select offspring by tournament selection
        offspring = [
            copy.deepcopy(individual)
            for individual in tools.selTournament(population, len(population), 2)
        ]

        # Apply genetic operations on the offspring
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            operations.mate(ind1, ind2)
            operations.mutate(ind1)
            operations.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values  # outdated

        # Evaluate new fitness values
        try:
            evaluator.evaluate(offspring)
        except BudgetError:
            if show_progress_bar:
                # Close progress bar properly if we exhaust the budget
                iterator.close()
            break

        # Select new population by choosing the highest UCB values
        population = tools.selBest(population + offspring, population_size)

        # Check diversity
        fraction_distinct = len({tuple(individual) for individual in population}) / len(population)

        # If we show a progress bar, update some information shown there
        if show_progress_bar:
            best_fitness = max(individual.fitness.values[0] for individual in population)
            iterator.set_postfix(
                OrderedDict((("fitness", best_fitness), ("distinct", fraction_distinct)))
            )

        # Abort optimization if the diversity is too low
        if fraction_distinct < min_diversity:
            if show_progress_bar:
                iterator.close()
            break

    # Find maximum
    # Note: The best individual must still be in the selection,
    # because we always keep the best individuals.
    best = max(population, key=lambda ind: ind.fitness.values[0])
    return best.to_sequence(wildtype), best.fitness.values[0]

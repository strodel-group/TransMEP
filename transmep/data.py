import copy
import csv
import re
from typing import List, Tuple

import fsspec
import numpy as np

# 20 amino acids plus unknown 'X'.
amino_acid_alphabet = list("RHKDESTNQCUGPAVILMFYWX")

# Utility file for loading and writing datasets.


class DatasetError(Exception):
    """
    Thrown if some error occurs during dataset parsing.
    """

    def __init__(self, message: str):
        """
        Create a new dataset error.

        :param message: Human-readable message.
        """
        self.message = message


class Dataset:
    """
    Main dataset object.
    A dataset always contains the wildtype sequence, all variant sequences,
    and unique IDs for all variants.
    It may also contain target values.
    """

    def __init__(
        self,
        wildtype: str,
        variants: np.ndarray,
        y: np.ndarray = None,
        ids: np.ndarray = None,
    ):
        """
        Create a new dataset.

        :param wildtype: Wildtype sequence.
        :param variants: Variant sequences as a NumPy array of strings.
        :param y: Optional target values.
        :param ids: IDs of the variants. If None, IDs are generated.
        """
        if y is not None:
            assert len(variants) == len(y)
        if ids is None:
            ids = np.arange(len(variants))
        else:
            assert len(variants) == len(ids)
        assert all(len(variant) == len(wildtype) for variant in variants)

        self.wildtype = wildtype
        self.variants = variants
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.variants)

    def __getitem__(self, item):
        """
        Can process arbitrary NumPy indexing.

        :param item: Indices as expected by NumPy.
        :return: The indexed dataset.
        """
        if self.y is not None:
            y = self.y[item]
        else:
            y = None
        return Dataset(self.wildtype, self.variants[item], y=y, ids=self.ids[item])


def load_wildtype(wildtype_path: str) -> str:
    """
    Load wild type sequence.

    :param wildtype_path: Path to the sequence. Supports remote paths via fsspec.
    :return: The wildtype sequence.
    """
    with fsspec.open(wildtype_path, "r") as fd:
        wildtype = fd.read().rstrip()
        if not all(c in amino_acid_alphabet for c in wildtype):
            raise DatasetError("Invalid amino acid letter in wild type sequence!")
    return wildtype


def load_dataset(wildtype_path: str, variants_path: str) -> Dataset:
    """
    Load a dataset.

    :param wildtype_path: Path to wild type sequence.  Supports remote paths via fsspec.
    :param variants_path: Path to variants CSV with columns 'mutations' and optionally 'y'.
         Supports remote paths via fsspec.
    :return: Dataset object.
    """
    wildtype = load_wildtype(wildtype_path)
    with fsspec.open(variants_path, "r") as fd:
        reader = csv.DictReader(fd)
        variants = []
        y = None
        for row in reader:
            variants.append(parse_mutations_notation(wildtype, row["mutations"].strip()))
            if "y" in row:
                if y is None:
                    y = []
                y.append(float(row["y"]))

    if y is not None:
        y = np.asarray(y, dtype=np.float32)  # TransMEP always uses float32.

    return Dataset(wildtype, np.asarray(variants, dtype=object), y=y)


def parse_mutations_notation(wildtype: str, mutant: str) -> str:
    """
    Parses a sequence in mutations notation, e.g. 'R34H+N64Q'.
    The wild type is expected to be denoted by ''.

    :param wildtype: Wild type sequence.
    :param mutant: Mutations notation.
    :return: The variant's sequence.
    """
    if len(mutant) == 0:
        return copy.deepcopy(wildtype)
    variant = list(wildtype)
    for mutation in mutant.split("+"):
        match = re.match("([RHKDESTNQCUGPAVILMFYWX])([0-9]+)([RHKDESTNQCUGPAVILMFYWX])", mutation)
        if not match:
            raise DatasetError("Invalid mutation '%s'!" % mutation)
        original = match.group(1)
        position = int(match.group(2)) - 1
        new = match.group(3)
        if original != wildtype[position]:
            raise DatasetError("Mutation %s does not match wild type sequence!" % mutation)
        variant[position] = new
    return "".join(variant)


def write_dataset(dataset: Dataset, path: str):
    """
    Write a dataset's variants to CSV.

    :param dataset: Dataset, y values are optional.
    :param path: Path to write to. Supports remote paths via fsspec.
    :return:
    """
    with fsspec.open(path, "w") as fd:
        fieldnames = ["mutations"]
        if dataset.y is not None:
            fieldnames.append("y")
        writer = csv.DictWriter(fd, fieldnames)
        writer.writeheader()
        for i in range(len(dataset)):
            entry = {"mutations": get_mutations_notation(dataset.wildtype, dataset.variants[i])}
            if dataset.y is not None:
                entry["y"] = dataset.y[i]
            writer.writerow(entry)


def get_mutations_notation(wildtype: str, variant: str) -> str:
    """
    Convert a sequence to mutations notation, e.g. 'R34H+N64Q'.
    The wild type is denoted by ''.

    :param wildtype: Wild type sequence.
    :param variant: Variant's sequence.
    :return: Mutations notation.
    """
    mutations = []
    for pos, (acid_wt, acid_variant) in enumerate(zip(wildtype, variant)):
        if acid_wt != acid_variant:
            mutations.append("%s%d%s" % (acid_wt, pos + 1, acid_variant))
    return "+".join(mutations)


def get_mutations(dataset: Dataset) -> List[Tuple[str, int, str]]:
    """
    Return all distinct mutations in a dataset by tuples (old_acid, position, new_acid).

    :param dataset: Dataset object.
    :return: List of tuples (old_acid, position, new_acid).
    """
    mutations = set()
    for variant in dataset.variants:
        for p, (a, b) in enumerate(zip(dataset.wildtype, variant)):
            if a != b:
                mutations.add((a, p, b))
    mutations = list(mutations)
    mutations.sort(key=lambda x: (x[1], x[2]))
    return mutations


def strip_y(dataset: Dataset) -> Dataset:
    """
    Remove target value information from a dataset.

    :param dataset: Dataset object (unchanged).
    :return: A new dataset object without target values.
    """
    return Dataset(dataset.wildtype, dataset.variants, ids=dataset.ids)

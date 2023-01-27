import copy
import csv
import re
from typing import List, Tuple

import fsspec
import numpy as np

amino_acid_alphabet = list("RHKDESTNQCUGPAVILMFYWX")


class DatasetError(Exception):
    def __init__(self, message: str):
        self.message = message


class Dataset:
    def __init__(
        self,
        wildtype: str,
        variants: np.ndarray,
        y: np.ndarray = None,
        ids: np.ndarray = None,
    ):
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
        if self.y is not None:
            y = self.y[item]
        else:
            y = None
        return Dataset(self.wildtype, self.variants[item], y=y, ids=self.ids[item])


def load_wildtype(wildtype_path: str) -> str:
    with fsspec.open(wildtype_path, "r") as fd:
        wildtype = fd.read().rstrip()
        if not all(c in amino_acid_alphabet for c in wildtype):
            raise DatasetError("Invalid amino acid letter in wild type sequence!")
    return wildtype


def load_dataset(wildtype_path: str, variants_path: str) -> Dataset:
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
        y = np.asarray(y, dtype=np.float32)

    return Dataset(wildtype, np.asarray(variants, dtype=object), y=y)


def parse_mutations_notation(wildtype: str, mutant: str) -> str:
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
    mutations = []
    for pos, (acid_wt, acid_variant) in enumerate(zip(wildtype, variant)):
        if acid_wt != acid_variant:
            mutations.append("%s%d%s" % (acid_wt, pos + 1, acid_variant))
    return "+".join(mutations)


def get_mutations(dataset: Dataset) -> List[Tuple[str, int, str]]:
    mutations = set()
    for variant in dataset.variants:
        for p, (a, b) in enumerate(zip(dataset.wildtype, variant)):
            if a != b:
                mutations.add((a, p, b))
    mutations = list(mutations)
    mutations.sort(key=lambda x: (x[1], x[2]))
    return mutations


def strip_y(dataset: Dataset) -> Dataset:
    return Dataset(dataset.wildtype, dataset.variants, ids=dataset.ids)

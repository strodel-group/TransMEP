from typing import Tuple, Union

import esm

FOUNDATION_MODELS = [
    "esm1b_t33_650M_UR50S",
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]


def load_foundation_model(
    foundation_model: str,
) -> Tuple[Union[esm.ProteinBertModel, esm.ESM2], esm.Alphabet]:
    foundation_model, alphabet = {
        "esm1b_t33_650M_UR50S": esm.pretrained.esm1b_t33_650M_UR50S,
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
        "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
    }[foundation_model]()
    return foundation_model, alphabet

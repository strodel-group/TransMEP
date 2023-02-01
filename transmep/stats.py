from typing import Callable, Tuple

import numpy as np
from arch.bootstrap import IIDBootstrap


def bootstrap(
    values: np.ndarray,
    metric: Callable = np.mean,
    repetitions: int = 1000,
    coverage: float = 0.95,
    method: str = "bca",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap a metric to estimate a confidence interval.

    :param values: Values to operate on.
    :param metric: Metric to bootstrap.
    :param repetitions: Bootstrap repetitions.
    :param coverage: Confidence level.
    :param method: Computation method.
    :return:
    """
    values = np.asarray(values)
    estimate = metric(values)
    if len(values) <= 1:
        return estimate, estimate, estimate
    bs = IIDBootstrap(np.asarray(values))
    lower, upper = bs.conf_int(metric, reps=repetitions, method=method, size=coverage)
    return lower, estimate, upper

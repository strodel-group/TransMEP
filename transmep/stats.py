import numpy as np
from arch.bootstrap import IIDBootstrap


def bootstrap(values, metric=np.mean, repetitions=1000, coverage=0.95, method="bca"):
    values = np.asarray(values)
    estimate = metric(values)
    if len(values) <= 1:
        return estimate, estimate, estimate
    bs = IIDBootstrap(np.asarray(values))
    lower, upper = bs.conf_int(metric, reps=repetitions, method=method, size=coverage)
    return lower, estimate, upper

from typing import Callable


def is_positive_int() -> Callable[[str], bool]:
    """
    Provides a callable that checks a string for whether it's a positive integer.

    :return: A callable from string to bool.
    """

    def inner(answer: str) -> bool:
        try:
            value = int(answer)
            return value > 0
        except ValueError:
            return False

    return inner


def is_percentage() -> Callable[[str], bool]:
    """
    Provides a callable that checks a string for whether it's a percentage.

    :return: A callable from string to bool.
    """

    def inner(answer: str) -> bool:
        try:
            value = float(answer) / 100
            return 0 < value < 1
        except ValueError:
            return False

    return inner

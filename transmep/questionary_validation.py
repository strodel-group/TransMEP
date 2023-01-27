from typing import Callable


def is_positive_int() -> Callable[[str], bool]:
    def inner(answer: str) -> bool:
        try:
            value = int(answer)
            return value > 0
        except ValueError:
            return False

    return inner


def is_percentage() -> Callable[[str], bool]:
    def inner(answer: str) -> bool:
        try:
            value = float(answer) / 100
            return 0 < value < 1
        except ValueError:
            return False

    return inner

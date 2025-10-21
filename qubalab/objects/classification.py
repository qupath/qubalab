from __future__ import annotations
import random
from typing import Optional, Union


class Classification(object):
    """
    Simple class to store the names and color of a classification.
    """

    _cached_classifications: dict[str, Classification] = {}

    def __new__(
        cls, names: Union[str, tuple[str]], color: Optional[tuple[int, int, int]] = None
    ):
        if isinstance(names, str):
            names = (names,)
        if isinstance(names, list):
            names = tuple(names)
        if names is None:
            return None
        if not isinstance(names, tuple):
            raise TypeError("names should be str or tuple[str]")
        name = ": ".join(names)
        classification = Classification._cached_classifications.get(name)
        if classification is None:
            classification = super().__new__(cls)
            Classification._cached_classifications[name] = classification
        return classification

    def __init__(
        self,
        names: Union[str, tuple[str]],
        color: Optional[tuple[int, int, int]] = None,
    ):
        """
        :param names: the names of the classification
        :param color: the RGB color (each component between 0 and 255) of the classification. Can be None to use a random color
        """
        if isinstance(names, str):
            names = (names,)
        elif isinstance(names, list):
            names = tuple(names)
        if not isinstance(names, tuple):
            raise TypeError("names should be a tuple, list or string")
        self._names = names
        self._color = (
            tuple(random.randint(0, 255) for _ in range(3)) if color is None else color
        )

    @property
    def name(self) -> str:
        """
        The name of the classification.
        """
        return ": ".join(self._names)

    @property
    def names(self) -> tuple[str]:
        """
        The name of the classification.
        """
        return self._names

    @property
    def color(self) -> tuple[int, int, int]:
        """
        The color of the classification.
        """
        return self._color  ## todo: pylance type hints problem

    def __str__(self):
        return f"Classification {self.name} of color {self.color}"

    def __repr__(self):
        return f"Classification('{self.name}', {self.color})"

    def __eq__(self, other):
        if isinstance(other, Classification):
            return (self is other) or (self.name == other.name)
        return False

    def __hash__(self):
        return hash(self.name)

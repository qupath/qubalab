from __future__ import annotations
import random
from typing import Optional, Union


class Classification(object):
    """
    Simple class to store the names and color of a classification.

    Each Classification with the same names is the same object, retrieved from a cache.
    Therefore updating the color of a Classification will update all similarly classified objects.
    """

    __cached_classifications: dict[tuple[str], Classification] = {}

    def __new__(
        cls, names: Union[str, tuple[str]], color: Optional[tuple[int, int, int]] = None
    ):
        if isinstance(names, str):
            names = (names,)
        elif isinstance(names, list):
            names = tuple(names)
        if names is None:
            return None
        if not isinstance(names, tuple):
            raise TypeError("names should be str or tuple[str]")

        classification = Classification.__cached_classifications.get(names)
        if classification is None:
            classification = super().__new__(cls)
            Classification.__cached_classifications[names] = classification

        if color is not None:
            classification.color = color
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
        self.__names = names
        self.__color: tuple = (
            tuple(random.randint(0, 255) for _ in range(3)) if color is None else color
        )

    @property
    def names(self) -> tuple[str]:
        """
        The names of the classification.
        """
        return self.__names

    @property
    def color(self) -> tuple[int, int, int]:
        """
        The color of the classification.
        """
        return self.__color  ## todo: pylance type hints problem

    @color.setter
    def color(self, value: tuple[int, int, int]) -> None:
        """
        Change the color of the classification.
        :param value: the new 8-bit RGB color
        """
        self.__color = value

    def __str__(self):
        return f"Classification {self.names} of color {self.color}"

    def __repr__(self):
        return f"Classification('{self.names}', {self.color})"

    def __eq__(self, other):
        if isinstance(other, Classification):
            return (self is other) or (self.names == other.names)
        return False

    def __hash__(self):
        return hash(self.names)

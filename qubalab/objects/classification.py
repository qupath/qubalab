from __future__ import annotations
import random
from typing import Optional, Union


class Classification(object):
    """
    Simple class to store the names and color of a classification.
    """

    _cached_classifications = {}

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
        self._names = names
        self._color = (
            tuple(random.randint(0, 255) for _ in range(3)) if color is None else color
        )

    @property
    def name(self) -> str:
        """
        The name of the classification.
        """
        if self._names is None:
            print("lol")
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

    @staticmethod
    def get_cached_classification(
        name: Optional[Union[str, tuple[str]]],
        color: Optional[tuple[int, int, int]] = None,
    ) -> Optional[Classification]:
        """
        Return a classification by looking at an internal cache.

        If no classification with the provided name is present in the cache, a
        new classification is created and the cache is updated.

        This is useful if you want to avoid creating multiple classifications with the
        same name and use only one instead.

        :param name: the name of the classification (can be None)
        :param color: the RGB color (each component between 0 and 255) of the classification.
                      Can be None to use a random color. This is only used if the cache doesn't
                      already contain a classification with the provided name
        :return: a classification with the provided name, but not always with the provided color
                 if a classification with the same name already existed in the cache. If the provided
                 name is None, None is also returned
        """
        if name is None:
            return None
        if isinstance(name, str):
            name = (name,)
        name = ": ".join(name)
        classification = Classification._cached_classifications.get(name)
        if classification is None:
            classification = Classification(name, color)
            Classification._cached_classifications[classification.name] = classification
        return classification

    def __str__(self):
        return f"Classification {self.name} of color {self.color}"

    def __repr__(self):
        return f"Classification('{self.name}', {self.color})"

    def __eq__(self, other):
        if isinstance(other, Classification):
            return self.name == other.name and self.color == other.color
        return False

    def __hash__(self):
        return hash(self.name)

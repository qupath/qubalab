from __future__ import annotations
import random


class Classification(object):
    """
    Simple class to store the name, color and parent of a classification.
    """

    def __init__(self, name: str, color: tuple[int, int, int] = None, parent: Classification = None):
        """
        Create the classification.

        :param name: the name of the classification
        :param color: the RGB color (each component between 0 and 255) of the classification. Can be None to use a random color
        :param parent: the parent of the classification. Can be None if the classification has no parent
        """
        self._parent = parent
        self._name = name
        self._color = [random.randint(0, 255) for _ in range(3)] if color is None else color

    @property
    def name(self) -> str:
        """
        The name of the classification.
        """
        return self._name

    @property
    def color(self) -> tuple[int, int, int]:
        """
        The color of the classification.
        """
        return self._color

    def __str__(self):
        return f"Classification {self._name} of color {self._color}"
        
    def __repr__(self):
        if self._parent is None:
            return f"Classification('{self._name}', {self._color})"
        else:
            return f"Classification('{self._name}', {self._color}, {repr(self._parent)})"
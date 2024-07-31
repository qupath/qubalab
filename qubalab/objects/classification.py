from __future__ import annotations
import random


class Classification(object):
    """
    Simple class to store the name and color of a classification.
    """
    _cached_classifications = {}

    def __init__(self, name: str, color: tuple[int, int, int] = None):
        """
        :param name: the name of the classification
        :param color: the RGB color (each component between 0 and 255) of the classification. Can be None to use a random color
        """
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
    
    @staticmethod
    def get_cached_classification(name: str, color: tuple[int, int, int] = None) -> Classification:
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
        
        classification = Classification._cached_classifications.get(name)
        if classification is None:
            classification = Classification(name, color)
            Classification._cached_classifications[classification.name] = classification
        return classification

    def __str__(self):
        return f"Classification {self._name} of color {self._color}"
        
    def __repr__(self):
        return f"Classification('{self._name}', {self._color})"
        
    def __eq__(self, other):
        if isinstance(other, Classification):
            return self._name == other._name and self._color == other._color
        return False
    
    def __hash__(self):
        return hash(self._name)

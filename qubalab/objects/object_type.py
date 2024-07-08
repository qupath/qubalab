from enum import Enum


class ObjectType(Enum):
    """
    Represent an object in QuPath.
    """
    ROOT = 1
    ANNOTATION = 2
    DETECTION = 3
    TILE = 4
    CELL = 5
    TMA_CORE = 6

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ImageShape:
    """
    Simple data class to store an image shape.

    Useful to avoid ambiguity about dimension order.
    TODO: This can potentially be removed if we use xarray instead of numpy arrays, and/or embrace AICSImageIO's
          dimension ordering standard.
    :param x: the image width
    :param y: the image height
    :param t: the number of time points
    :param c: the number of channels
    :param z: the number of z-stacks
    """
    x: int
    y: int
    t: int = 1
    c: int = 1
    z: int = 1

    """
    Create an ImageShape from a list of arguments.
    
    There must be at least five arguments, in the following order: tczyx.
    :returns: an ImageShape corresponding to the arguments
    :raises IndexeError: when there are less than five arguments
    """
    def from_tczyx(*args) -> ImageShape:
        return ImageShape(t=args[0], c=args[1], z=args[2], y=args[3], x=args[4])

    """
    Return a tuple describing this ImageShape.

    :param dims: the format the resulting tuple should have. Each character must be one of 'tczyx'
    :returns: a tuple describing this ImageShape with the specified format
    :raises AttributeError: when a character of dims is not in 'tczyx'
    """
    def as_tuple(self, dims: str = 'tczyx') -> tuple:
        return tuple(self.__getattribute__(d) for d in dims)
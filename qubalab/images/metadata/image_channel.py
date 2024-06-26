from dataclasses import dataclass


@dataclass(frozen=True)
class ImageChannel:
    """
    Simple data class to store the name and the color of a channel of an image.

    :param name: the name of the channel
    :param color: the RGB color of the channel, with each component between 0 and 1
    """
    name: str
    color: tuple[float, float, float]

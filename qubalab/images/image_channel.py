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


_DEFAULT_CHANNEL_SINGLE = (
    ImageChannel(name='Single channel', color=(1, 1, 1)),
)

_DEFAULT_CHANNEL_RGB = (
    ImageChannel(name='Red', color=(1, 0, 0)),
    ImageChannel(name='Green', color=(0, 1, 0)),
    ImageChannel(name='Green', color=(0, 0, 1)),
)

_DEFAULT_CHANNEL_TWO = (
    ImageChannel(name='Channel 1', color=(1, 0, 1)),
    ImageChannel(name='Channel 2', color=(0, 1, 0))
)

_DEFAULT_CHANNEL_COLORS = (
    (0, 1, 1),
    (1, 1, 0),
    (1, 0, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1)
)
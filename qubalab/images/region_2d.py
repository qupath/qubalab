from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Region2D:
    """
    Simple data class to represent the bounding box for a region in 2D.

    :param scale_factor: optional downsample value to indicate that the region may be required at a lower resolution.
                         The downsample is only applied to the x and y coordinates
    :param x: the top left x-coordinate of the bounding box
    :param y: the top left y-coordinate of the bounding box
    :param width: the width of the bounding box
    :param height: the height of the bounding box
    :param z: the z-slice index of the bounding box
    :param t: the time point index of the bounding box
    """
    downsample: float = None
    x: int = 0
    y: int = 0
    width: int = -1
    height: int = -1
    z: int = 0
    t: int = 0

    def scale_region(self, scale_factor: float = None) -> Region2D:
        """
        Scale the bounding box for the Region2D.

        :param scale_factor: the scale factor to apply to this region. Can be omitted to use the downsample of this region
        :returns: this Region2D scaled by the provided factor
        :raises ValueError: when scale_factor is 0
        """
        return self.downsample_region(None if scale_factor is None else 1.0 / scale_factor)

    def downsample_region(self, downsample: float = None) -> Region2D:
        """
        Downsample the bounding box for the Region2D.

        This can be used to convert coordinates from (for example) the full image resolution 
        to a different pyramidal level.
        :param downsample: the downsample to apply to this region. Can be omitted to use the downsample of this region
        :returns: this Region2D downsampled by the provided factor
        :raises ValueError: when downsample is 0
        """
        if downsample is None:
            downsample = self.downsample

        if downsample is None or downsample == 1:
            return self

        if downsample == 0:
            raise ValueError('Downsample cannot be 0!')

        x = int(self.x / downsample)
        y = int(self.y / downsample)
        # Handle -1 for width & height, i.e. until the full image width
        if self.width == -1:
            x2 = x - 1
        else:
            x2 = int(round(self.x + self.width) / downsample)
        if self.height == -1:
            y2 = y - 1
        else:
            y2 = int(round(self.y + self.height) / downsample)
        return Region2D(downsample=None, x=x, y=y, width=x2 - x, height=y2 - y, z=self.z, t=self.t)
    
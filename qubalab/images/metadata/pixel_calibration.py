from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PixelLength:
    """
    Simple data class to store pixel size information, along one dimension.

    Can be thought of as the pixel width, pixel height or pixel depth (z-spacing).
    
    :param length: the length of the pixel, by default 1
    :param unit: a text describing the unit of length, by default "pixels"
    """
    length: float = 1.0
    unit: str = 'pixels'

    def is_default(self) -> bool:
        """
        Returns True if this is a default value (length is 1.0 and unit is 'pixels')

        :returns: whether this is a default pixel length
        """
        return self.length == 1.0 and self.unit == 'pixels'

    @staticmethod
    def create_microns(length: float) -> PixelLength:
        """
        Create a PixelLength with a unit of micrometers (µm).

        :param length: the length of the pixel
        :returns: a pixel length of the provided length with the 'micrometer' unit
        """
        return PixelLength(length=length, unit='micrometer')

    @staticmethod
    def create_unknown(length: float) -> PixelLength:
        """
        Create a PixelLength with an unknown unit.

        :param length: the length of the pixel
        :returns: a pixel length of the provided length with no unit
        """
        return PixelLength(length=length, unit=None)


@dataclass(frozen=True)
class PixelCalibration:
    """
    Simple data class for storing pixel calibration information.

    :param length_x: the pixel size along the x-axis
    :param length_y: the pixel size along the y-axis
    :param length_z: the pixel size along the z-axis
    """
    length_x: PixelLength = PixelLength()
    length_y: PixelLength = PixelLength()
    length_z: PixelLength = PixelLength()

    def is_calibrated(self) -> bool:
        """
        Indicate if this PixelCalibration has at least one non-default length.

        :returns: whether this PixelCalibration has at least one non-default length
        """
        for size in [self.length_x, self.length_y, self.length_z]:
            if not size.is_default():
                return True
        return False

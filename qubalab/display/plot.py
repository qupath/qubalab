import matplotlib.pyplot as plt
import numpy as np


def plotImage(image, title=None):
    """
    A utility function to display a NumPy array returned by an ImageServer.

    :param image: a numpy array with dimensions (c, y, x). If c == 3, all channels will be displayed.
                  Otherwise, only the first channel will be displayed
    :param title: the title of the plot. Can be omitted if no title should be displayed
    """
    
    # Move channel axis
    image = np.moveaxis(image, [0], [2])

    # Keep only first channel if there are more than 3 channels
    if image.shape[2] > 3:
        image = image[..., 0]

    # If pixel values are float numbers, they can be greater than 0, while matplotlib only accepts
    # values between 0 and 1, so the array is normalized between 0 and 1
    if image.dtype != "uint8":
        image /= image.max()

    if title is not None:
        plt.title(title)

    plt.imshow(image)
    plt.show()
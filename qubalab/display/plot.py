import matplotlib.pyplot as plt
import numpy as np


def plotImage(image, title=None, channel=0):
    """
    A utility function to display a NumPy array returned by an ImageServer.

    :param image: a numpy array with dimensions (c, y, x). If it represents a RGB image, all channels will be displayed.
                  Otherwise, only the specified channel will be displayed
    :param title: the title of the plot. Can be omitted if no title should be displayed
    :param channel: the channel to display if the image is not RGB
    """
    
    # Move channel axis
    image = np.moveaxis(image, [0], [2])

    # Keep only specified channel if the image is not RGB
    if image.shape[2] != 3 or image.dtype != "uint8":
        image = image[..., channel]

    # If pixel values are float numbers, they can be greater than 0, while matplotlib only accepts
    # values between 0 and 1, so the array is normalized between 0 and 1
    if "float" in str(image.dtype):
        image /= image.max()

    if title is not None:
        plt.title(title)

    plt.imshow(image)
    plt.axis(False)
    plt.show()
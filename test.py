import dask.array as da
import dask
import math
import numpy as np


display = False
image = np.random.rand(3, 10, 20)
width = image.shape[-1]
height = image.shape[-2]
channels = image.shape[-3]
dtype = image.dtype

chunk_size = (3, 6, 6)
chunk_width = chunk_size[-1]
chunk_height = chunk_size[-2]


def read_chunk(i, j):
    if display:
        print("Entered: " + str(i) + "/" + str(j))
    return image[..., chunk_height*j:chunk_height*(j+1), chunk_width*i:chunk_width*(i+1)]


arrays = []
for i in range(math.ceil(width / chunk_width)):
    column = []
    for j in range(math.ceil(height / chunk_height)):
        column.append(da.from_delayed(
            dask.delayed(read_chunk)(i, j),
            shape=(
                channels,
                min(chunk_height, height - chunk_height*j),
                min(chunk_width, width - chunk_width*i),
            ),
            dtype=dtype
        ))
    arrays.append(da.concatenate(column, axis=1))
    
array = da.concatenate(arrays, axis=2)
print(array.shape)

np.testing.assert_array_equal(array.compute(), image)
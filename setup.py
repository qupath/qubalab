from setuptools import setup

requirements = [
    "dask[array]>=2021.4.1",
    "zarr>=2.11.0",
    "numpy",
    "Shapely>=1.8.4",
    "imageio",
    "tiffslide",
    "imagecodecs==2021.4.28"
#    "matplotlib"
]

setup(
    name="QuPyth",
    version="0.0.1-snapshot",
    author="Pete Bankhead",
    description="A laboratory for exploring quantitative bioimage analysis in Python",
    long_description="Not QuPath... but very friendly with QuPath",
    install_requires=requirements,
    extras_require={
        "openslide": ["openslide-python"],
        "qupath": ["py4j>=0.10.9.0"]
    }
)
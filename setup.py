from setuptools import setup

requirements = [
    "dask[array]",
    "zarr",
    "numpy",
    "Shapely",
    "geojson",
    "imageio",
    "tiffslide",
    "aicsimageio",
    "matplotlib"
]

setup(
    name="QuBaLab",
    version="0.0.1.dev0",
    author="Pete Bankhead",
    description="A laboratory for exploring quantitative bioimage analysis in Python",
    long_description="QuBaLab isn't QuPath... but it's very friendly with QuPath",
    install_requires=requirements,
    extras_require={
        "openslide": ["openslide-python"],
        "qupath": ["py4j>=0.10.9.0"],
        "test": ["pytest"]
    }
)
# QuBaLab

This is a Python package for exploring quantitative bioimage analysis... *especially* (but not exclusively) in 
combination with QuPath (https://qupath.github.io/).

The name comes from **Quantitative Bioimage Analysis Laboratory**.
This is chosen to be reminiscent of QuPath (*Quantitative Pathology*), but recognizes that neither is really restricted
to pathology.

## Goals

QuBaLab isn't QuPath - they're just good friends.

* **QuPath** is a user-friendly Java application for bioimage analysis, which has some especially nice features for
handling whole slide and highly-multiplexed images. But lots of bioimage analysis researcher is done in Python,
and is hard to integrate with QuPath.
* **QuBaLab**'s main aim is to help with this, by providing tools to help exchange data between QuPath and Python
*without any direct dependency on QuPath and Java*. It therefore doesn't require QuPath to be installed, and
can be used entirely from Python.

QuBaLab doesn't share code with QuPath, but is uses many of the same conventions for accessing images and
representing objects in a GeoJSON-compatible way.
By using the same custom fields for things like measurements and classifications, exchanging data is much easier.

## Getting started

You can find the documentation on https://qupath.github.io/qubalab/.

This project contains the QuBaLab package in the `qubalab` folder. Take a look at the *Installation* section to install it.

Some notebooks present in the `notebooks` folder show how to use the QuBaLab package. If you want to run them, you can take a look at the *Development* section.
If you just want to go through them, look at the [documentation](https://qupath.github.io/qubalab/notebooks.html). 

## Installation

TODO when available on PyPI

Installing the package through PyPI will only give access to the Python library. If you want to run the notebooks or
contribute to this project, take a look at the *Development* section.

## Development

This part is useful if you want to run the notebooks or contribute to this project.

You will have to install [Poetry](https://python-poetry.org/docs/#installation).

Then, run:

```bash
git clone https://github.com/qupath/qubalab.git         # to clone this repository
cd qubalab                                              # to go to the project directory
poetry install            # to install the dependencies
poetry run jupyter-lab    # to start the Jupyter notebooks
poetry run pytest         # to run unit tests
```

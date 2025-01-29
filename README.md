# QuBaLab

[![Coverage](https://qupath.github.io/badges/qubalab/badges/coverage-badge.svg?dummy=1234)](https://qupath.github.io/badges/qubalab/reports/coverage/index.html?dummy=1234)
[![Tests](https://qupath.github.io/badges/qubalab/badges/tests-badge.svg?dummy=1234)](https://qupath.github.io/badges/qubalab/reports/junit/report.html?dummy=1234)
![Actions](https://github.com/qupath/qubalab/actions/workflows/tests.yml/badge.svg?dummy=1234)

> ## ⚠️ Important
>
> * 🆕  **This is a new Python package for interacting with QuPath**!
> * 💻 **For a stable, established solution, check out [paquo](https://github.com/Bayer-Group/paquo/) (although we aim to meet different needs, see below)**
> * 🐛 **Please report any bugs through the [issues page](https://github.com/qupath/qubalab/issues)**

This is a Python package for exploring quantitative bioimage analysis... *especially* (but not exclusively) in combination with [QuPath](https://qupath.github.io/).

The name comes from **Quantitative Bioimage Analysis Laboratory**. This is chosen to be reminiscent of QuPath (*Quantitative Pathology*), but recognizes that neither is really restricted to pathology.

## Why use QuBaLab?

QuBaLab isn't QuPath - they're just good friends.

* **QuPath** is a user-friendly Java application for bioimage analysis, which has some especially nice features for handling whole slide and highly-multiplexed images. But lots of bioimage analysis research is done in Python, and is hard to integrate with QuPath.
* **QuBaLab**'s main aim is to help with this, by providing tools to help exchange data between QuPath and Python *without any direct dependency on QuPath and Java*. It therefore doesn't require QuPath to be installed, and can be used entirely from Python.

QuBaLab doesn't share code with QuPath, but is uses many of the same conventions for accessing images and representing objects in a GeoJSON compatible way. By using the same custom fields for things like measurements and classifications, exchanging data is much easier.

### How does QuBaLab compare to paquo?

[paquo](https://paquo.readthedocs.io/) is an existing library linking Python and QuPath that provides a pythonic interface to QuPath.

*We think paquo is great - we don't want to replace it!*

Here are the 3 main differences as we see them:

1. **Target audience**
    * paquo is written mostly for Python programmers who need to work with QuPath data
    * QuBaLab is written mostly for QuPath users who want to dip into Python
2. **Convenience vs. Efficiency**
    * paquo is based on [JPype](http://jpype.readthedocs.io/) to provide full & efficient access to Java from Python
    * QuBaLab is based on [Py4J](https://www.py4j.org) to exchange data between Java & Python - preferring convenience to efficiency
3. **Pixel access**
    * paquo is for working with QuPath projects and objects - accessing pixels is beyond its scope (at least for now)
    * QuBaLab enables requesting pixels as numpy or dask arrays, and provides functions to convert between thresholded images & QuPath objects

So if you're a Python programmer who needs an intuitive and efficient way to work with QuPath data, use paquo.

But if you're a QuPath user who wants to switch to Python for some tasks, including image processing, you might want to give QuBaLab a try.

## Getting started

You can find the documentation on [github pages](https://qupath.github.io/qubalab-docs/).

This project contains the QuBaLab package in the `qubalab` folder. Take a look at the *Installation* section to install it.

Some notebooks in the `notebooks` folder demonstrate how to use QuBaLab. If you want to run them, you can take a look at the *Development* section. If you just want to browse the content in them, look at the [documentation](https://qupath.github.io/qubalab-docs/notebooks.html).

## Installation

QuBaLab is on PyPI, so you can install it with pip:

```bash
pip install --upgrade qubalab
```

## Development

This part is useful if you want to run the notebooks or contribute to this project.

First, run:

```bash
git clone https://github.com/qupath/qubalab.git         # clone this repository
cd qubalab                                              # go to the project directory
python -m venv ./.venv                                  # create a local virual environment
source ./.venv/bin/activate                             # activate the venv
pip install -e ".[dev,test]"                  # install qubalab (-e means changes are loaded dynamically)
jupyter lab .                                           # to start the Jupyter notebooks
pytest                                                  # to run unit tests
```

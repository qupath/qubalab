# QuBaLab

This is a Python package for exploring quantitative bioimage analysis... *especially* (but not exclusively) in 
combination with QuPath (https://qupath.github.io/).

The name comes from **Quantitative Bioimage Analysis Laboratory**.
This is chosen to be reminiscent of QuPath (*Quantitative Pathology*), but recognizes that neither is really restricted
to pathology.

> QuBaLab is at a very early stage of development, and is likely to change quickly!
> It's not yet ready for general use, but if you're interested in contributing, please get in touch.

## Goals

QuBaLab isn't QuPath - they're just good friends.

QuPath is a user-friendly Java application for bioimage analysis, which has some especially nice features for handling 
whole slide and highly-multiplexed images.

But lots of bioimage analysis researcher is done in Python, and is hard to integrate with QuPath.

QuBaLab's main aim is to help with this, by providing tools to help exchange data between QuPath and Python *without 
any direct dependency on QuPath and Java*.
It therefore doesn't require QuPath to be installed, and can be used entirely from Python.

QuBaLab doesn't share code with QuPath, but is uses many of the same conventions for accessing images and representing 
objects in a GeoJSON-compatible way.
By using the same custom fields for things like measurements and classifications, exchanging data is much easier.


## Installation

QuBaLab is not yet available on PyPI, so I'm afraid you'll need to do things manually.

Instructions will be added here whenever I or someone else figures out what they are.

## Development

Install poetry
poetry run pytest to run tests
notebooks
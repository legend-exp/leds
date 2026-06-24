# Configuration file for the Sphinx documentation builder.
from __future__ import annotations

import sys
from importlib.metadata import version as _version
from pathlib import Path

sys.path.insert(0, Path(__file__).parents[2].resolve().as_posix())

project = "leds"
copyright = "Copyright Holder"
version = _version("leds")

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "python"

# Furo theme
html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/legend-exp/leds",
    "source_branch": "main",
    "source_directory": "docs/source",
}
html_title = f"{project} {version}"

# Heavy/scientific dependencies mocked at doc-build time: autodoc only needs to
# read leds' own signatures and docstrings, and importing these for real pulls
# in the lgdo -> awkward-pandas -> pandas chain, which can be mutually
# incompatible in the slim docs environment.
autodoc_mock_imports = [
    "awkward",
    "awkward_pandas",
    "bokeh",
    "dbetto",
    "dspeed",
    "legendmeta",
    "lgdo",
    "lh5",
    "matplotlib",
    "mplhep",
    "pandas",
    "panel",
    "scipy",
]  # add new packages here
autodoc_default_options = {"ignore-module-all": True}

# sphinx-napoleon
# enforce consistent usage of NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}  # add new intersphinx mappings here

# sphinx-autodoc
# Include __init__() docstring in class docstring
autoclass_content = "both"
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented_params"
autodoc_typehints_format = "short"

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('../bin'))
sys.path.insert(0, os.path.abspath('../detectron2-0.4_mod'))
print(sys.path)

# -- Project information -----------------------------------------------------

project = 'pcnaDeep'
copyright = '2021, Chan Kuan Yoow Group'
author = 'Chan Kuan Yoow Group'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "rst2pdf.pdfbuilder",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

# -- Configurations for plugins ------------
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = True
napoleon_use_rtype = True
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"
pdf_documents = [('index', u'rst2pdf', u'Test', u'Jeff')]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.7", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
# -------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

letex_engine = 'xelatex'
latex_elements = {
    'papersize': 'a4paper',
    # Additional stuff for the LaTeX preamble.
    'preamble':r'''
    \usepackage{xeCJK}
    \usepackage{indentfirst}
    \setlength{\parindent}{2em}
    \setCJKmainfont[BoldFont=STFangsong, ItalicFont=STKaiti]{STSong}
    \setCJKsansfont[BoldFont=STHeiti]{STXihei}
    \setCJKmonofont{STFangsong}
    ''',
}




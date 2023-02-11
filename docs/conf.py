# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from datetime import date
import sphinx_rtd_theme

from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform

from polaris.version import __version__

# -- Project information -----------------------------------------------------

project = "Polaris"
copyright = f"{date.today().year}, Energy Exascale Earth System Model Project"
author = "E3SM Development Team"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
if 'DOCS_VERSION' in os.environ:
    version = os.environ.get('DOCS_VERSION')
    release = version
else:
    # The short X.Y.Z version.
    version = __version__
    # The full version, including alpha/beta/rc tags.
    release = __version__

master_doc = "index"
language = "en"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx_multiversion",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

autosummary_generate = True

# Otherwise, the Return parameter list looks different from the Parameters list
napoleon_use_rtype = False
# Otherwise, the Attributes parameter list looks different from the Parameters
# list
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None),
    'geometric_features':
        ('http://mpas-dev.github.io/geometric_features/stable/', None),
    'mpas_tools':
        ('http://mpas-dev.github.io/MPAS-Tools/stable/', None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

# -- Autodoc settings ---------------------------------------------------

autodoc_member_order = "bysource"
nitpicky = True
nitpick_ignore = [
    ("py:class", "docutils.nodes.document"),
    ("py:class", "docutils.nodes.docinfo"),
    ("py:class", "docutils.nodes.Element"),
    ("py:class", "docutils.nodes.Node"),
    ("py:class", "docutils.nodes.field_list"),
    ("py:class", "docutils.nodes.problematic"),
    ("py:class", "docutils.nodes.pending"),
    ("py:class", "docutils.nodes.system_message"),
    ("py:class", "docutils.statemachine.StringList"),
    ("py:class", "docutils.parsers.rst.directives.misc.Include"),
    ("py:class", "docutils.parsers.rst.Parser"),
    ("py:class", "docutils.utils.Reporter"),
    ("py:class", "nodes.Element"),
    ("py:class", "nodes.Node"),
    ("py:class", "nodes.system_message"),
    ("py:class", "Directive"),
    ("py:class", "Include"),
    ("py:class", "StringList"),
    ("py:class", "DocutilsRenderer"),
    ("py:class", "MockStateMachine"),
    ("py:exc", "MarkupError"),
]

# -- MyST settings ---------------------------------------------------

myst_enable_extensions = []
myst_number_code_blocks = ["typescript"]
myst_heading_anchors = 2
myst_footnote_transition = True
myst_dmath_double_inline = True
myst_enable_checkboxes = True

# -- HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_title = ""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

smv_tag_whitelist = r"^v\d+\.\d+.\d+$"  # Include tags like "tags/v2.5.0"
smv_branch_whitelist = "main"
smv_remote_whitelist = r"^(origin|upstream)$"  # Use branches from origin

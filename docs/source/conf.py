# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MedClip'
copyright = '2023, Daniel Vela'
author = 'Daniel Vela'
release = '1.0'
#-- Adding code paths to system path-----------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_toolbox.collapse',
    'sphinx.ext.doctest'
]

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_logo = "mindkind_logo.png"
html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "sidebar_hide_name": False,
}

# Autodoc configuration---------------------------------------------------------
autodoc_default_options = {

    'special-members': '__init__',

}

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#05757a",
        "color-brand-content": "#05757a"
    },
}


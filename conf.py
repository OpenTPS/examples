# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OpenTPS'
copyright = '2024, OpenTPS team'
author = 'OpenTPS team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_gallery.gen_gallery',
]


sphinx_gallery_conf = {
    # Path to your examples directory
    'examples_dirs': 'examples',  # Path to your example scripts
    'gallery_dirs': 'auto_examples',  # Path where the gallery will be generated
    'filename_pattern': '.*',  # Pattern to match example files
    'nested_sections': True,  # Allow nested sections in the examples
    #"ignore_pattern": "SimpleOptimizationProton|SimpleOptimizationPhoton|boundConstraintsOpti|simpleOptimization_createDicomStudy|PlanDeliverySimulation",

}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
templates_path = ['_templates']

html_theme_options = {
    'repository_url': 'https://github.com/OpenTPS/examples',
    "article_header_end": "my_header.html",
    "show_nav_level": 2,
}
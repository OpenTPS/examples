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


from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    # Path to your examples directory
    'examples_dirs': 'examples',  # Path to your example scripts
    'gallery_dirs': 'auto_examples',  # Path where the gallery will be generated
    'within_subsection_order': FileNameSortKey,
    # Add a Colab badge
    'binder': {
        'org': 'OpenTPS',
        'repo': 'examples/blob',  # Replace with your GitHub repository
        'branch': 'main',
        'binderhub_url': 'https://colab.research.google.com/github',
        'dependencies': 'requirements.txt',
        'notebooks_dir': 'notebooks',
        'use_jupyter_lab': False,
    },
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

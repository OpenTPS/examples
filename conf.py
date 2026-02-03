# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OpenTPS examples'
copyright = '2024, OpenTPS team'
author = 'OpenTPS team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_gallery.gen_gallery',
]


sphinx_gallery_conf = {
    "examples_dirs": [
        "examples",             # your main examples
        "community",   # new folder for community contributions
    ],
    "gallery_dirs": [
        "auto_examples",        # output folder for main gallery
        "auto_community",    # output folder for community gallery
    ],
    'filename_pattern': r'run_.*\.py$',  # matches any file starting with 'run_'
    'ignore_pattern': [r'__init__\.py',r'examples/showStuff.py',r'examples/syntheticData.py'],   # files to exclude from gallery generation
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
templates_path = ['_templates']

html_logo = "_static/OpenTPS_logo_dark_big.png"   # path relative to docs folder

html_theme_options = {
    'repository_url': 'https://github.com/OpenTPS/examples',
    "article_header_end": "my_header.html",
    "show_nav_level": 2,
    "logo": {
        "text": "OpenTPS examples",  # text next to the logo
        "image_light": "_static/OpenTPS_logo_dark_big.png",  # logo for light mode
    },
}
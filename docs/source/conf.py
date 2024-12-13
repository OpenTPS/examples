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

extensions = ['sphinx_gallery.gen_gallery','sphinx_togglebutton','sphinx_examples']

sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_theme_options = { "repository_url": "https://github.com/OpenTPS/examples",
                       "repository_branch": "master",
                       "path_to_docs": "docs",
                        "launch_buttons":{
                            "binderhub_url": "https://mybinder.org",
                            "colab_url": "https://colab.research.google.com/",
                            "deepnote_url": "https://deepnote.com/",
                            "notebook_interface": "jupyterlab",
                        },
                       "use_issues_button": True,
                       "use_repository_button": True,
                       "use_download_button": True,
                       "use_fullscreen_button":False,
                       }


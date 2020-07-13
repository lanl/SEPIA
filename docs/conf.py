import sys
import os

sys.path.insert(0, os.path.abspath('..'))

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'sphinx.ext.autosectionlabel']
source_suffix = '.rst'
master_doc = 'index'
project = 'Sepia'
exclude_patterns = ['_build']
pygments_style = 'sphinx'
html_theme = 'bizstyle'

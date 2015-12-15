try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Non-IID Ridge Regression using CRFs.',
    'author': 'Nico Goernitz',
    'url': 'https://github.com/nicococo/niidbox',
    'download_url': 'https://github.com/nicococo/niidbox',
    'author_email': 'nico.goernitz@tu-berlin.de',
    'version': '0.1',
    'install_requires': ['cvxopt','numba'],
    'packages': ['niidbox'],
    'scripts': [],
    'name': 'niidbox'
}

setup(**config)
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Non-IID Ridge Regression with latent (discrete) dependencies between examples.',
    'url': 'https://github.com/nicococo/niidbox',
    'author': 'Nico Goernitz',
    'author_email': 'nico.goernitz@tu-berlin.de',
    'version': '0.1',
    'install_requires': ['nose', 'cvxopt', 'numba'],
    'packages': ['niidbox'],
    'scripts': [],
    'name': 'niidbox',
    'classifiers':['Intended Audience :: Science/Research',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7']
}

setup(**config)
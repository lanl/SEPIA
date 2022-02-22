import setuptools
from setuptools import setup

setup(name='sepia',
      version='1.1',
      description='Gaussian process emulation and calibration',
      url='http://www.github.com/lanl/SEPIA',
      author='James Gattiker, Natalie Klein, Grant Hutchings, Earl Lawrence',
      author_email='',
      license='BSD',
      packages=setuptools.find_packages(),
      zip_safe=False,
      python_requires='>=3.8',
      install_requires=[
            'numpy',
            'pandas',
            'matplotlib',
            'scipy',
            'seaborn',
            'statsmodels',
            'tqdm'
            ]
      )



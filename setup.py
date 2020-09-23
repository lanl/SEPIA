from setuptools import setup

setup(name='sepia',
      version='1.1',
      description='Gaussian process emulation and calibration',
      url='http://www.github.com/lanl/SEPIA',
      author='James Gattiker, Natalie Klein, Grant Hutchings, Earl Lawrence',
      author_email='',
      license='BSD',
      packages=['sepia'],
      zip_safe=False,
      install_requires=[
            'numpy',
            'matplotlib',
            'scipy',
            'seaborn',
            'statsmodels',
            'tqdm'
            ]
      )



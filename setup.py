from setuptools import setup, find_packages

setup(
    name='clustermol',
    version='1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/wenkanglu/clustermol',
    license='',
    author='Wen Kang Lu, Nicholas Limbert, Robyn Mckenzie',
    author_email='lxxwen005@myuct.ac.za',
    description='A framework for processing and clustering MD trajectories.',
    entry_points={
              'console_scripts': [
                  'clustermol = main.clustermol:main'
              ]
          },
    install_requires=[
        'hdbscan',
        'matplotlib',
        'mdtraj',
        'numpy',
        'pandas',
        'seaborn',
        'scipy',
        'sklearn',
        'umap-learn',
    ],
)

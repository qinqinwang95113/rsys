import numpy, os, glob

from setuptools import Command, Extension, setup

from Cython.Build import cythonize
from Cython.Compiler.Options import _directive_defaults

_directive_defaults['linetrace'] = True
_directive_defaults['binding'] = True
compile_args = ['-Wno-unused-function', '-Wno-maybe-uninitialized', '-Wno-sign-compare', '-O3', '-ffast-math', '-fopenmp']
link_args = ['-fopenmp']

# Look for cython files to compile
cython_files = []
for path in glob.glob("recsys_framework/**/*.pyx", recursive=True):
    cython_files.append((path.replace(".pyx", "").replace(os.sep, "."), path))

modules = [Extension(file[0], [file[1]], language='c++', extra_compile_args=compile_args, extra_link_args=link_args)
           for file in cython_files]
setup(
    name='recsys_framework',
    version="1.0.0",
    description='recommender System Framework',
    url='https://github.com/MaurizioFD/RecSysFramework.git',
    author='Cesare Bernardis, Maurizio Ferrari Dacrema',
    author_email='cesare.bernardis@polimi.it, maurizio.ferrari@polimi.it',
    install_requires=['numpy>=0.14.0',
                      'pandas>=0.22.0',
                      'scipy>=0.16',
                      'scikit-learn>=0.19.1',
                      'matplotlib>=3.0',
                      'Cython>=0.27',
                      'nltk>=3.2.5',
                      'similaripy>=0.0.11'],
    packages=['recsys_framework',
              'recsys_framework.evaluation',
              'recsys_framework.data_manager',
              'recsys_framework.data_manager.reader',
              'recsys_framework.data_manager.dataset_postprocessing',
              'recsys_framework.data_manager.splitter',
              'recsys_framework.recommender',
              'recsys_framework.recommender.GraphBased',
              'recsys_framework.recommender.knn',
              'recsys_framework.recommender.MatrixFactorization',
              'recsys_framework.recommender.SLIM',
              'recsys_framework.recommender.SLIM.BPR',
              'recsys_framework.recommender.SLIM.ElasticNet',
              'recsys_framework.parameter_tuning',
              'recsys_framework.utils',
              'recsys_framework.utils.Similarity',
              ],
    setup_requires=["Cython >= 0.27"],
    ext_modules=cythonize(modules),
    include_dirs=[numpy.get_include()]
)
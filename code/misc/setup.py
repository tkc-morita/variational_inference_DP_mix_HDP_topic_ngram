#coding: utf-8

from distutils.core import setup
from distutils.extension import Extension

from distutils.core import setup
from Cython.Build import cythonize
import numpy

# setup(
#     cmdclass = {'build_ext': build_ext},
#     ext_modules = [Extension("simulation_uniform", ["simulation_uniform.pyx"], include_dirs = [numpy.get_include()])]
# )

setup(
	ext_modules = cythonize("*.pyx"),
	include_dirs=[numpy.get_include()]
	)
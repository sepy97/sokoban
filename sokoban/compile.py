from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

from numpy import get_include
import os

ext = [
       Extension("sokoban_interface", ["sokoban_interface.pyx", "sokoban.cpp"],
                 include_dirs=[get_include()],
                 extra_compile_args=["-Ofast", "-march=native", "-std=c++14", '-funroll-loops'],
                 # extra_link_args=['-fopenmp']
                 language="c++")
       ]

setup(ext_modules=cythonize(ext, annotate=True), cmdclass={'build_ext': build_ext})
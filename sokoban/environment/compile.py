from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext = [
       Extension("sokoban_interface", ["sokoban_interface.pyx", "sokoban.cpp"],
                 extra_compile_args=["-Ofast", "-march=native", "-std=c++14"],
                 extra_link_args=['-fopenmp'],
                 language="c++")
       ]

setup(ext_modules=cythonize(ext, annotate=True), cmdclass={'build_ext': build_ext})
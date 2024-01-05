import sys

assert sys.version_info >= (3, 8, 0), "Support Python >= 3.8"


from setuptools import setup, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.extension import Extension

from numpy import get_include

MODULE_NAME = "wormtracer"


ext = Extension(
    "{}.skeleton".format(MODULE_NAME),
    sources=["./src/wormtracer/skeleton.pyx"],
    include_dirs=[get_include()],
)

# install setup
setup(
    ext_modules=cythonize([ext]),
    include_dirs=[get_include()],
)

import os
import platform
import subprocess

import numpy
from wheel.bdist_wheel import bdist_wheel
from setuptools import setup, Extension

class bdist_wheel_abi3(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            # on CPython, our wheels are abi3 and compatible back to 3.6
            return "cp36", "abi3", plat

        return python, abi, plat

# Thanks:
# https://github.com/joerick/python-abi3-package-sample


# From:
# https://stackoverflow.com/questions/60174152/how-do-i-add-pkg-config-the-setup-py-of-a-cython-wrapper
def pkgconfig(package, kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    output = subprocess.getoutput(
        'pkg-config --cflags --libs {}'.format(package))
    for token in output.strip().split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw

# We hardcode these; That's the default location that the "make install" by
# SuiteSparse; chooses. No luck using pkg-config :(
_suitesparse_components = [
    'SPQR', 'CHOLMOD', 'AMD', 'CAMD', 'COLAMD', 'CCOLAMD', 'SuiteSparseConfig']
if platform.system() == "Windows":
    kw = {
        'include_dirs': [
            f"C:\\Program Files (x86)\\{component}\\include\\suitesparse"
            for component in _suitesparse_components],
        'library_dirs': [ # unclear if the DLL's are used at build time
            f"C:\\Program Files (x86)\\{component}\\bin"
            for component in _suitesparse_components]
            + [f"C:\\Program Files (x86)\\{component}\\lib"
               for component in _suitesparse_components],
        # this is super important; compiling with MinGW we get lib*.dll files
        # if switch to MSVC then use "cholmod" and "spqr"
        'libraries': ["libcholmod", "libspqr"]}
else:
    kw = {'include_dirs': [], 'library_dirs': [], 'libraries': []}
kw['include_dirs'].append(numpy.get_include())
pkgconfig('SPQR', kw)
pkgconfig('CHOLMOD', kw)

setup(
    packages = ["pyspqr"],
    ext_modules =  [
        Extension(
            "_pyspqr",
            sources=['_pyspqr.c'],
            include_dirs=kw['include_dirs'],
            library_dirs=kw['library_dirs'],
            libraries=kw['libraries'],
            extra_compile_args=["-O3", "-fopenmp-simd"],
            # We define it in the code for safety
            # define_macros=[("Py_LIMITED_API", "0x03060000")],
            py_limited_api = True
        )
    ],
    cmdclass={"bdist_wheel": bdist_wheel_abi3},
)

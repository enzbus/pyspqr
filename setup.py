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

# we pick up Conda SuiteSparse on Windows, in GitHub CI
# see here https://docs.conda.io/projects/conda-build/en/latest/resources/use-shared-libraries.html
if ("CONDA_PREFIX" in os.environ) and (platform.system() == "Windows"):
    kw = {
        'include_dirs':[f"{os.environ['CONDA_PREFIX']}\\Library\\include\\suitesparse"],
        'library_dirs':[
            f"{os.environ['CONDA_PREFIX']}\\Library\\lib",
            f"{os.environ['CONDA_PREFIX']}\\Library\\bin"],
        'libraries':["cholmod", "spqr"]}
else:
    kw = {'include_dirs':[], 'library_dirs':[], 'libraries':[]}
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
            extra_compile_args=[],
            # We define it in the code for safety
            # define_macros=[("Py_LIMITED_API", "0x03060000")],
            py_limited_api = True
        )
    ],
    cmdclass={"bdist_wheel": bdist_wheel_abi3},
)

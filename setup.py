import subprocess

import numpy
from setuptools import setup, Extension

# From:
# https://stackoverflow.com/questions/60174152/how-do-i-add-pkg-config-the-setup-py-of-a-cython-wrapper
def pkgconfig(package, kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    output = subprocess.getoutput(
        'pkg-config --cflags --libs {}'.format(package))
    for token in output.strip().split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw

kw = {'include_dirs':[], 'library_dirs':[], 'libraries':[]}
kw['include_dirs'].append(numpy.get_include())
pkgconfig('SPQR', kw)
pkgconfig('CHOLMOD', kw)


setup_args = dict(
    ext_modules = [
        Extension(
            "_suitesparseqr",
            sources=['_suitesparseqr.c', ],
            **kw,
            extra_compile_args=[],
            py_limited_api = True
        )
    ]
)
setup(**setup_args)
import sys
import os
from os.path import splitext, basename
from glob import glob
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np


SUPERPOWERED_ROOT = os.environ.get('SUPERPOWERED_ROOT', '')

libraries = {
    'darwin': ['SuperpoweredAudioOSX'],
    'linux2': ['SuperpoweredLinuxX86_64'],
}

extra_link_args = {
    'darwin': ['-framework', 'AVFoundation']
}

ext_modules = [
    Extension(
        'superpowered.%s' % splitext(basename(file))[0],
        [file],
        language='c++',
        library_dirs=[SUPERPOWERED_ROOT],
        libraries=libraries.get(sys.platform),
        extra_link_args=extra_link_args.get(sys.platform),
        include_dirs=[SUPERPOWERED_ROOT, np.get_include()],
    )
    for file in glob('superpowered/*.pyx')
]

setup(
    name='superpowered',
    version='1.0.0',
    description='Python Wrapper for Superpowered SDK',
    author='John Snyder',
    author_email='johncsnyder@gmail.com',
    # url='http://github.com/cmcqueen/simplerandom',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    long_description=open('readme.md').read(),
    # test_suite='nose.collector',
    # tests_require=['nose'],
)
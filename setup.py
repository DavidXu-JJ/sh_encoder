"""
borrow from nr3d_lib by Jianfei Guo
"""
import os
import sys
import logging
import subprocess
from copy import deepcopy
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

common_library_dirs = []
# NOTE: On cluster machines's login node etc. where no GPU is installed,
#           `libcuda.so` will not be in usally place,
#           so `-lcuda` will raise "can not find -lcuda" error.
#       To solve this, we can use `libcuda.so` in the `stubs` directory.
# https://stackoverflow.com/questions/62999715/when-i-make-darknet-with-cuda-1-usr-bin-ld-cannot-find-lcudaoccured-how
if '--fix-lcuda' in sys.argv:
    sys.argv.remove('--fix-lcuda')
    common_library_dirs.append(os.path.join(os.environ.get('CUDA_HOME'), 'lib64', 'stubs'))

library_dirs = deepcopy(common_library_dirs)

major, minor = torch.cuda.get_device_capability()
compute_capability = major * 10 + minor

nvcc_flags = [
    '-O3', '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
    f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
]

if os.name == "posix":
    c_flags = ["-std=c++17"]
    nvcc_flags += [
        "-Xcompiler=-mf16c",
        "-Xcompiler=-Wno-float-conversion",
        "-Xcompiler=-fno-strict-aliasing",
        "-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
    ]
elif os.name == "nt":
    c_flags = ["/std:c++17"]

ext = CUDAExtension(
        name='sh_encoder._shencoder', # extension name, import this to use CUDA API
        sources=[os.path.join(SCRIPT_DIR, 'shencoder', f) for f in [
            'shencoder.cu',
            'bindings.cpp',
        ]],
        extra_compile_args={
            'cxx': c_flags,
            'nvcc': nvcc_flags,
        },
        library_dirs=library_dirs
)

setup(
    name="sh_encoder",
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension}
)

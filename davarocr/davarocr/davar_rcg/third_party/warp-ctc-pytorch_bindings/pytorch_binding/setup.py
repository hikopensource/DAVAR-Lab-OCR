import os
import platform
import sys
from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch

extra_compile_args = ['-std=c++14', '-fPIC']
warp_ctc_path = "../build"

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

if "WARP_CTC_PATH" in os.environ:
    warp_ctc_path = os.environ["WARP_CTC_PATH"]
if not os.path.exists(os.path.join(warp_ctc_path, "libwarpctc" + lib_ext)):
    print(("Could not find libwarpctc.so in {}.\n"
           "Build warp-ctc and set WARP_CTC_PATH to the location of"
           " libwarpctc.so (default is '../build')").format(warp_ctc_path))
    sys.exit(1)

include_dirs = [os.path.realpath('../include')]

if torch.cuda.is_available() or "CUDA_HOME" in os.environ:
    enable_gpu = True
else:
    print("Torch was not built with CUDA support, not building warp-ctc GPU extensions.")
    enable_gpu = False

if enable_gpu:
    from torch.utils.cpp_extension import CUDAExtension

    build_extension = CUDAExtension
    extra_compile_args += ['-DWARPCTC_ENABLE_GPU']
else:
    build_extension = CppExtension

ext_modules = [
    build_extension(
        name='warpctc_pytorch._warp_ctc',
        language='c++',
        sources=['src/binding.cpp'],
        include_dirs=include_dirs,
        library_dirs=[os.path.realpath(warp_ctc_path)],
        libraries=['warpctc'],
        extra_link_args=['-Wl,-rpath,' + os.path.realpath(warp_ctc_path)],
        extra_compile_args=extra_compile_args
    )
]

setup(
    name="warpctc_pytorch",
    version="0.1",
    description="PyTorch wrapper for warp-ctc",
    url="https://github.com/baidu-research/warp-ctc",
    author="Jared Casper, Sean Naren",
    author_email="jared.casper@baidu.com, sean.narenthiran@digitalreasoning.com",
    license="Apache",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)

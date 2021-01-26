from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mylinear_cuda',
    ext_modules=[
        CUDAExtension('mylinear_cuda', [
            'mylinear_cuda.cpp',
            'mylinear_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
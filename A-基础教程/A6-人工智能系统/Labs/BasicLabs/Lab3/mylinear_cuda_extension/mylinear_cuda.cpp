// MIT License

// Copyright (c) Microsoft Corporation.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

#include <torch/extension.h>

#include <iostream>
#include <vector>

// CUDA funciton declearition
std::vector<torch::Tensor> mylinear_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights);

std::vector<torch::Tensor> mylinear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights); 

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> mylinear_forward(
    torch::Tensor input,
    torch::Tensor weights) 
{
    CHECK_INPUT(input);
    CHECK_INPUT(weights);

    return mylinear_cuda_forward(input, weights);
}

std::vector<torch::Tensor> mylinear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights) 
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weights);

    return mylinear_cuda_backward(grad_output, input, weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mylinear_forward, "myLinear forward (CUDA)");
  m.def("backward", &mylinear_backward, "myLinear backward (CUDA)");
}

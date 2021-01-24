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

std::vector<torch::Tensor> mylinear_forward(
    torch::Tensor input,
    torch::Tensor weights) 
{
    auto output = torch::mm(input, weights.transpose(0, 1));
    
    return {output};
}

std::vector<torch::Tensor> mylinear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights
    ) 
{
    auto grad_input = torch::mm(grad_output, weights);
    auto grad_weights = torch::mm(grad_output.transpose(0, 1), input);

    return {grad_input, grad_weights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mylinear_forward, "myLinear forward");
  m.def("backward", &mylinear_backward, "myLinear backward");
}

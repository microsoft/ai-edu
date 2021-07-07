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

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void matmul_kernel(
    const scalar_t* A,
    const scalar_t* B,
    scalar_t* C,
    const int M, 
    const int K, 
    const int N,
    const bool trans_A = false,
    const bool trans_B = false) 
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N)
    {
        scalar_t sum = 0.0;
        for (int k = 0; k < K; k++)
        {
            const int i = trans_A ? (k * M + row) : (row * K + k);
            const int j = trans_B ? (col * K + k) : (k * N + col);
            sum += A[i] * B[j];
        }

        C[row * N + col]  = sum;
    }
}

std::vector<torch::Tensor> mylinear_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights)
{
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weights.size(0);

    auto output = torch::zeros({M, N}, torch::TensorOptions().device(torch::kCUDA));

    const dim3 block(32, 32);
    const dim3 grid((M - 1) / 32 + 1, (N - 1) / 32 + 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_forward", ([&] {
        matmul_kernel<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            weights.data<scalar_t>(),
            output.data<scalar_t>(),
            M,
            K,
            N,
            false,
            true);
        }));
    
    return {output};
}

std::vector<torch::Tensor> mylinear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights)
{
    const int M = grad_output.size(0);
    const int N = grad_output.size(1);
    const int K = weights.size(1);

    auto grad_input = torch::zeros({M, K}, torch::TensorOptions().device(torch::kCUDA));
    auto grad_weights = torch::zeros({N, K}, torch::TensorOptions().device(torch::kCUDA));

    const dim3 block(32, 32);
    const dim3 grid1((M - 1) / 32 + 1, (K - 1) / 32 + 1);
    const dim3 grid2((N - 1) / 32 + 1, (K - 1) / 32 + 1);


    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_backward_input", ([&] {
        matmul_kernel<scalar_t><<<grid1, block>>>(
            grad_output.data<scalar_t>(),
            weights.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            M,
            N,
            K,
            false,
            false);
        }));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_backward_input", ([&] {
        matmul_kernel<scalar_t><<<grid2, block>>>(
            grad_output.data<scalar_t>(),
            input.data<scalar_t>(),
            grad_weights.data<scalar_t>(),
            N,
            M,
            K,
            true,
            false);
        }));
    
    return {grad_input, grad_weights};
}
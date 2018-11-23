relu(x) = max(x, 0)
sigmoid(x) = 1 / (1 + exp(-1 * x))
softmax(x) = exp.(x) ./ sum(exp.(x))
# parameters initialize methods
glorot_normal(dims...) = Float32.(randn(dims...) .* sqrt(2.0/sum(dims)))

# the definition for Dense_m
mutable struct Dense_m{F,A,V}
    σ::F
    weight::A
    bias::V
    regularizationw::Bool # indicate whether to regularize the weight
    regularizationb::Bool # indicate whether to regularize the bias
    input
    output
end


# two initialize methods
function Dense_m(in::Integer, out::Integer, σ = identity; initw = glorot_normal, initb = zeros)
    return Dense_m(σ, initw(out, in), initb(out), false, false, nothing, nothing)
end

function Dense_m(in::Integer, out::Integer, regularizationw::Bool, regularizationb::Bool, σ = identity; initw = glorot_normal, initb = zeros)
    return Dense_m(σ, initw(out, in), initb(out), regularizationw, regularizationb, nothing, nothing)
end

# the forward method for Dense
function (d::Dense_m)(x)
    x = reshape(x, (:, size(x)[end]))
    σ, w, b = d.σ, d.weight, d.bias
    d.output = σ.(w * x .+ b)
    d.input = x
    return d.output
end

# the definition for Conv
mutable struct Conv_m{F,B,C,V}
    σ::F
    weight::B
    bias::C
    stride::Int
    outputChannels::V
    padding::Bool
    input
    output
    colImage
end

# initialize for Conv, padding=true used for the same size output, weight_size is (width, height, in_channels, out_channels)
# to do: not support padding=false
function Conv_m(weight_size::AbstractArray{Int, 1}, stride=1, σ=relu; padding=true, init=glorot_normal)
    W, H, InputChannels, OutputChannels = weight_size
    return Conv_m(σ, init(H, W, InputChannels, OutputChannels), init(1, OutputChannels), stride, OutputChannels, padding, nothing, nothing, nothing)
end

# pad the input when the padding is true, now assume stride=1
function pad(s, len)
    W, H, C, B = size(s)
    paddingS = zeros(W + 2 * len, H + 2 * len, C, B)
    paddingS[(len + 1):W+len, (len + 1):W+len, :, :] = s
    return paddingS
end

# img2col method
function expand(image, kernelSize, strideSize)
    expandLength = (fld(size(image, 1) - kernelSize, strideSize) + 1) ^ 2
    expandWidth = kernelSize ^ 2 * size(image)[end]
    colImage = zeros(expandLength, expandWidth)
    for i in range(1; stop=size(image, 1) - kernelSize + 1, step=strideSize)
        baseIndex = (i - 1) * (fld(size(image, 1) - kernelSize, strideSize) + 1)
        for j in range(1; stop=size(image, 1) - kernelSize + 1, step=strideSize)
            colImage[baseIndex + j, :] = reshape(image[i:i + kernelSize - 1, j:j + kernelSize - 1,:], length(image[i:i+kernelSize-1, j:j+kernelSize-1,:]))
        end
    end
    return colImage
end

# the forward method of Conv
function (c::Conv_m)(x::AbstractArray{Float64, 4})
    if c.padding
        padding_size = fld(size(c.weight, 1), 2)
        input = pad(x, padding_size)
    else
        input = x
    end

    # record the initial input
    c.input = x

    input_shape = size(input)
    weights_shape = size(c.weight)
    output_shape = (fld(input_shape[1] - weights_shape[1], c.stride) + 1, fld(input_shape[2] - weights_shape[2], c.stride) + 1, weights_shape[4], input_shape[4])

    weights = reshape(c.weight, (:, c.outputChannels))

    c.output = zeros(output_shape)

    # for each img, do the computation
    expand_images = []
    for i in range(1; stop=input_shape[4])
        colImage = expand(input[:,:,:,i], weights_shape[1], c.stride)
        c.output[:, : , :, i] = reshape(colImage * weights .+ c.bias, output_shape[1:3])
        push!(expand_images, colImage)
    end

    c.colImage = expand_images
    c.output = c.σ.(c.output)
    return c.output

end

# copy from Flux, to inital the Chain struct
using MacroTools: @forward
import Adapt: adapt

struct Chain_m
  layers::Vector{Any}
  Chain_m(xs...) = new([xs...])
end

@forward Chain_m.layers Base.getindex, Base.first, Base.last, Base.lastindex, Base.push!
@forward Chain_m.layers Base.iterate

adapt(T, c::Chain_m) = Chain_m(map(x -> adapt(T, x), c.layers)...)

(c::Chain_m)(x) = foldl((x, m) -> m(x), c.layers; init = x)

Base.getindex(c::Chain_m, i::AbstractArray) = Chain_m(c.layers[i]...)

# the function used for the gradient computation
function gradient(network, loss)
    global batch_size
    len = length(network.layers)
    n = len
    grad_w = []
    grad_b = []

    # for each layer, loop
    while n > 0
        input, output = network[n].input, network[n].output
        loss = reshape(loss, size(output))

        # compute according to the activation function
        if network[n].σ == sigmoid
            loss = loss .* (output) .* (1 - output)
        elseif network[n].σ == relu
            loss = loss .* (output .> 0)
        end

        # backward according to the typr of each layer
        if isa(network[n], Dense_m)
            push!(grad_w, loss * transpose(input) ./ batch_size)
            push!(grad_b, reshape(sum(loss,dims=2), size(network[n].bias)) ./ batch_size)
            loss = transpose(network[n].weight) * loss

        elseif isa(network[n], Conv_m)

            weights = copy(network[n].weight)
            weights_shape = size(weights)
            input_shape = size(network[n].input)
            output_shape = size(network[n].output)
            preloss = reshape(loss, output_shape)
            loss = reshape(loss, (:, network[n].outputChannels, output_shape[4]))
            strideSize = network[n].stride
            batch_size = output_shape[4]
            weights_grad = 0

            # compute the grad for weights and bias
            for i in range(1;stop=batch_size)
                weights_grad = weights_grad .+ reshape((transpose(network[n].colImage[i]) * loss[:, :, i]), weights_shape)
            end
            push!(grad_w, weights_grad)
            bias_grad = reshape(sum(loss, dims=(1,3)), size(network[n].bias))
            push!(grad_b, bias_grad)

            # compute the backward loss
            preloss = pad(preloss, fld(weights_shape[1], 2))
            weights = weights[end:-1:1,end:-1:1,end:-1:1,end:-1:1]
            weights = permutedims(weights, (1,2,4,3))
            weights = reshape(weights, (:, size(weights, 4)))
            backloss = zeros(input_shape)
            for i in range(1; stop=input_shape[end])
                backloss[:, :, :, i] = reshape(
                expand(preloss[:, :, :, i], weights_shape[1], strideSize) * weights, input_shape[1:3]
                )
            end

            loss = backloss
        end
        n = n - 1
    end

    return (grad_w, grad_b)
end

# update each layers parameters
function update!(network, grad_w, grad_b)
    learning_rate = 0.005
    for layer in network.layers
        Δw = pop!(grad_w)
        Δb = pop!(grad_b)
        layer.weight -= Δw .* learning_rate
        layer.bias -= Δb .* learning_rate
    end
end


function backward(network, result, label)

    for i in range(1; length=size(result, 2))
        result[:, i] = softmax(result[:, i])
    end

    loss = result

    for i in range(1; length=size(loss, 2))
        loss[label[i] + 1, i] -= 1
    end

    grad_w, grad_b = gradient(network, loss)

    update!(network, grad_w, grad_b)
    return loss
end

# data preparation
using Flux.Data.MNIST

imgs = MNIST.images()
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...)

labels = MNIST.labels()

# the network
m = Chain_m(
  Conv_m([3,3,1,2]),
  Dense_m(28 ^2 * 2, 32, true, true, relu),
  Dense_m(32, 10, true, true))

epoch = 5
batch_size = 30
len = epoch * fld(size(X, 2), batch_size)
i = 1
Frequency = 100

# reshape imgs to the desired size
tempImgs = zeros((28,28,1,length(imgs)))
for i in range(1; stop=length(imgs))
    tempImgs[:, :, :, i] = Float32.(reshape(float.(imgs[i]),(28,28,1)))
end

# train loop
@time while i < len
    global i
    start = (1 + (i - 1) * batch_size) % size(X, 2)
    # tempX = zeros((28, 28, 1, batch_size))
    # for i in range(1; stop=batch_size)
    #     tempX[:, :, :, i] = Float32.(reshape(tempImg[i], (28,28, 1)))
    # end
    tempX = tempImgs[:, :, :, start:start + batch_size - 1]
    tempL = labels[start:start + batch_size - 1]
    loss = backward(m, m(tempX), tempL)
    i += 1
end

# test
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, throttle
using Base.Iterators: repeated

accuracy(x, y) = mean(onecold(softmax(m(x))) .== onecold(y))
# tX = hcat(float.(reshape.(MNIST.images(:test), :))...)
test = MNIST.images(:test)
tX = zeros((28, 28, 1, length(test)))
for i in range(1;stop=size(tX, 4))
    tX[:, :, :, i] = Float32.(reshape(test[i], (28,28,1)))
end

tY = onehotbatch(MNIST.labels(:test), 0:9)

print(accuracy(tX, tY))

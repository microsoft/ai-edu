relu(x) = max(x, 0)
sigmoid(x) = 1 / (1 + exp(-1 * x))
softmax(x) = exp.(x) ./ sum(exp.(x))

mutable struct Dense_m{F,A,V}
    σ::F
    weight::A
    bias::V
    regularizationw::Bool # indicate whether to regularize the weight
    regularizationb::Bool # indicate whether to regularize the bias
    input
    output
end

glorot_normal(dims...) = randn(dims...) .* sqrt(2.0/sum(dims))

# two initialize methods
function Dense_m(in::Integer, out::Integer, σ = identity; initw = glorot_normal, initb = zeros)
    return Dense_m(σ, initw(out, in), initb(out), false, false, nothing, nothing)
end

function Dense_m(in::Integer, out::Integer, regularizationw::Bool, regularizationb::Bool, σ = identity; initw = glorot_normal, initb = zeros)
    return Dense_m(σ, initw(out, in), initb(out), regularizationw, regularizationb, nothing, nothing)
end

function (d::Dense_m)(x)
    σ, w, b = d.σ, d.weight, d.bias
    d.output = σ.(w * x .+ b)
    d.input = x
    return d.output
end

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

function gradient(network, loss)
    len = length(network.layers)
    n = len
    grad_w = []
    grad_b = []
    while n > 0
        input, output = network[n].input, network[n].output
        if network[n].σ == sigmoid
            loss = loss .* (output) .* (1 - output)
        elseif network[n].σ == relu
            loss = loss .* (output .> 0)
        end
        push!(grad_w, loss * transpose(input) ./ batch_size)
        push!(grad_b, reshape(sum(loss,dims=2), size(network[n].bias)) ./ batch_size)
        loss = transpose(network[n].weight) * loss
        n = n - 1
    end

    return (grad_w, grad_b)
end

function update!(network, grad_w, grad_b)
    learning_rate = 0.005
    for layer in network.layers
        Δw = pop!(grad_w)
        Δb = pop!(grad_b)
        layer.weight -= Δw .* learning_rate
        layer.bias -= Δb .* learning_rate
    end
end
temp = []
flag = true
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


using Flux.Data.MNIST

imgs = MNIST.images()
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...)

labels = MNIST.labels()

m = Chain_m(
  Dense_m(28^2, 392, true, true, relu),
  Dense_m(392, 64, true, true, relu),
  Dense_m(64, 32, true, true, relu),
  Dense_m(32, 10, true, true))

epoch = 20
batch_size = 30
len = epoch * size(X, 2) // batch_size
i = 1
Frequency = 100
# using Profile
while i < len
    global i
    start = (1 + (i - 1) * batch_size) % size(X, 2)
    tempX = X[:, start:start + batch_size - 1]
    tempL = labels[start:start + batch_size - 1]
    loss = backward(m, m(tempX), tempL)
    i += 1
end

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, throttle
using Base.Iterators: repeated

accuracy(x, y) = mean(onecold(softmax(m(x))) .== onecold(y))
tX = hcat(float.(reshape.(MNIST.images(:test), :))...)
tY = onehotbatch(MNIST.labels(:test), 0:9)

print(accuracy(tX, tY))

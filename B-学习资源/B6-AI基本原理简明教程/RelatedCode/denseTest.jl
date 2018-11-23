using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, throttle
using Base.Iterators: repeated

# using CuArrays

# Classify MNIST digits with a simple multi-layer-perceptron
function crossentropy_modify(ŷ::AbstractVecOrMat, y::AbstractVecOrMat, m; weight = 1)
  loss1 = -sum(y .* log.(ŷ) .* weight) / size(y, 2)
  # for i in m
  #   len = 1
  #   for j in ndims(i)
  #     len = len * size(i, j)
  #   end
  #   loss1 = loss1 + 0.005 * sum(i .* i) / len
  # end
  return loss1
end

function collection_regularization(m::Chain)
  re = []
  for layer in m
    try
      if layer.regularizationw
        push!(re, layer.weight)
      end
      if i.regularizationb
        push!(re, layer.bias)
      end
    catch
      continue
    end
  end
  return re
end

imgs = MNIST.images()
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...)

labels = MNIST.labels()
# One-hot-encode the labels
Y = onehotbatch(labels, 0:9)

m = Chain(
  Dense_m(28^2, 32, true, true, relu),
  Dense_m(32, 10, true, true),
  softmax)

# get all paramaters will be regularized
re = collection_regularization(m)

# compute the loss and the regularization
loss(x, y) = crossentropy_modify(m(x), y, re)

# get the accuracy
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

# data_img = repeated(X, 20)
# data_img = collect(data_img)
# data_label = repeated(Y, 20)
# data_label = collect(data_label)
data_img = X
data_label = Y
evalcb = () -> @show(loss(X, Y))
opt = ADAM(params(m))
# data_set = (hcat(data_img...), hcat(data_label...))
data_set = (data_img, data_label)

epoch = 20
batch_size = 30
len = epoch * size(X, 2) // batch_size
i = 1
@time while i < len
  global i
  start = (1 + (i - 1) * batch_size) % size(data_img, 2)
  # temp_data = repeated((data_set[1][:, offset:offset+batch_size], data_set[2][:, offset:offset+batch_size]),1)
  temp_data = repeated((data_set[1][:, start:start + batch_size - 1], data_set[2][:, start:start + batch_size - 1]), 1)
  Flux.train!(loss, temp_data, opt, cb = throttle(evalcb, 10))
  i = i + 1
end

accuracy(X, Y)

# Test set accuracy
tX = hcat(float.(reshape.(MNIST.images(:test), :))...)
tY = onehotbatch(MNIST.labels(:test), 0:9)

print(accuracy(tX, tY))

using BSON
using BSON: @save

@save "out.model" m

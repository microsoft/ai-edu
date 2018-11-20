WORK_DIRECTORY = nothing
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 2
VALIDATION_SIZE = 500  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
TRAINING_SIZE = 12665
TESTING_SIZE = 2115
SOURCE_FOLDER = "./input"

train_data_filename = "./input/train-images-01"
train_labels_filename = "./input/train-labels-01"
test_data_filename = "./input/test-images-01"
test_labels_filename = "./input/test-labels-01"


# for name in (:train_data_filename, :train_labels_filename, :test_data_filename. :test_labels_filename)
#     eval(quote
#     $name = SOURCE_FOLDER * $name
#     end)
# end
print("function")
glorot_normal(dims...) = randn(dims...) .* sqrt(2.0/sum(dims))

function extract_image_data(filename::String, num_images)
    open(filename, "r") do f
        bin_data = read(f)
        bin_data = bin_data[17:length(bin_data)]
        data = Float64.(bin_data)
        data = (data .- (PIXEL_DEPTH / 2.0)) ./ PIXEL_DEPTH
        data = reshape(data, (num_images, IMAGE_SIZE, IMAGE_SIZE))
        return data
    end

end

function extract_labels_data(filename, num_images)
    open(filename, "r") do f
        bin_data = read(f)
        bin_data = bin_data[9:length(bin_data)]
        data = Int64.(bin_data)
        return data
    end

end

function initialize_array(num_input, nn_hidden, nn_output)
    W1 = glorot_normal(nn_hidden, num_input)
    B1 = zeros(nn_hidden, 1)
    W2 = glorot_normal(nn_output, nn_hidden)
    B2 = zeros(nn_output, 1)

    dict_params = Dict(
                "W1" => W1,
                "B1" => B1,
                "W2" => W2,
                "B2" => B2
                )
    return dict_params
end

function image2vector(image)
    one_image_vector = reshape(image, (784, 1))
    return one_image_vector
end

function normalize_data(one_image_vector)
    a_max = maximum(one_image_vector)
    a_min = minimum(one_image_vector)
    norm_one_image_vector = zeros(size(one_image_vector))
    for j in range(1, length=size(one_image_vector, 1))
        norm_one_image_vector[j]=(one_image_vector[j]-a_min)/(a_max-a_min)
    end
    return norm_one_image_vector
end

function sigmoid(x)
    y = 1 / (1 + exp.(-1 .* x))
    return y
end

function softmax(x)
    y = argmax(x)[1] - 1
    return y
end

function forward_calculation(X, dict_params)
    w1 = dict_params["W1"]
    b1 = dict_params["B1"]
    w2 = dict_params["W2"]
    b2 = dict_params["B2"]

    Z1 = w1 * X .+ b1

    A1 = tanh.(Z1)

    Z2 = w2 * A1 .+ b2

    A2 = sigmoid.(Z2)

    dict_cache = Dict(
                "Z1" => Z1,
                "A1" => A1,
                "Z2" => Z2,
                "A2" => A2
                )
    return A2, dict_cache
end

function getCost(A2, Y)
    t = 0.000001
    part1 = Y .* log2.(A2 .+ t)
    part2 = log2.(1 .+ t .- A2) .* (1 .- Y)
    part3 = sum(part1 .+ part2, dims=1)
    cost = -1 .* part3 / size(A2, 1)
    return cost
end

function back_propagation(dict_params, dict_cache, X, Y)
    W1 = dict_params["W1"]
    W2 = dict_params["W2"]

    A1 = dict_cache["A1"]
    A2 = dict_cache["A2"]
    Z1 = dict_cache["Z1"]

    dZ2 = A2-Y
    dW2 = dZ2 * transpose(A1)
    dB2 = sum(dZ2, dims=1)

    dZ1 = (transpose(W2) * dZ2) - (1 .- A1 .* A1)
    dW1 = dZ1 * transpose(X)
    dB1 = sum(dZ1, dims=1)

    dict_grads = Dict(
        "dW1" => dW1,
        "dB1" => dB1,
        "dW2" => dW2,
        "dB2" => dB2
    )
    return dict_grads
end

function update_params(dict_params, dict_grads, learning_rate)
    W1 = dict_params["W1"]
    B1 = dict_params["B1"]
    W2 = dict_params["W2"]
    B2 = dict_params["B2"]

    dW1 = dict_grads["dW1"]
    dB1 = dict_grads["dB1"]
    dW2 = dict_grads["dW2"]
    dB2 = dict_grads["dB2"]

    # 梯度下降的实际实现
    W1=W1.-learning_rate.*dW1
    B1=B1.-learning_rate.*dB1
    W2=W2.-learning_rate.*dW2
    B2=B2.-learning_rate.*dB2

    dict_params = Dict(
                "W1" => W1,
                "B1" => B1,
                "W2" => W2,
                "B2" => B2
                )
    return dict_params
end

function Test(test_images, test_labels, dict_params)
    ii = 0
    test_loop = size(test_images, 1)
    for i in range(1, length=test_loop)
        img_train = test_images[i, :, :]
        # vector_image = normalize_data(image2vector(img_train))
        vector_image = image2vector(img_train)
        label_trainx = test_labels[i]
        aa2, xxx = forward_calculation(vector_image, dict_params)
        predict_value = softmax(aa2)
        if predict_value == Int(label_trainx)
            ii = ii + 1
        end
    end
    return ii
end


train_data = extract_image_data(train_data_filename, TRAINING_SIZE)
train_labels = extract_labels_data(train_labels_filename, TRAINING_SIZE)
test_data = extract_image_data(test_data_filename, TESTING_SIZE)
test_labels = extract_labels_data(test_labels_filename, TESTING_SIZE)

validation_data = train_data[1 : VALIDATION_SIZE, :, :]
validation_labels = train_labels[1 : VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE + 1 : size(train_data, 1), :, :]
train_labels = train_labels[VALIDATION_SIZE + 1 : size(train_labels, 1)]
print("success")
# input data dimension
num_input=28*28
# neuron numbers in hidden layer
nn_hidden=10
# output layer neuron numbers
nn_output=2

dict_params=initialize_array(num_input,nn_hidden,nn_output)

loop = size(train_data, 1)
dict_params=initialize_array(num_input,nn_hidden,nn_output)

for epoch in range(1, length=20)
    global dict_params
    for i in range(1, length=loop)
        img_train = train_data[i, :, :]
        label_train1 = train_labels[i]
        label_train = zeros(nn_output, 1)
        ttt = 1
        if i > 1000
            ttt = ttt * 0.999
        end
        label_train[Int(train_labels[i]) + 1] = 1
        vector_image = image2vector(img_train)
        # norm_image = normalize_data(vector_image)
        norm_image = vector_image
        A2, dict_cache = forward_calculation(norm_image, dict_params)
        prelabel = softmax(A2)
        cost = getCost(A2, label_train)
        dict_grads = back_propagation(dict_params, dict_cache, vector_image, label_train)
        dict_params = update_params(dict_params, dict_grads, ttt)
        # print(dict_grads["dW1"][1][1])
        # print("\n")
        dict_grads["dW1"].=0
        dict_grads["dW2"].=0
        dict_grads["dB1"].=0
        dict_grads["dB2"].=0
        # print("cost after iteration $i")
        # print(cost)
        # print("\n")
    end

    right = Test(test_data, test_labels, dict_params)
    print("the right precision after $epoch is $right")
    print("\n")
end

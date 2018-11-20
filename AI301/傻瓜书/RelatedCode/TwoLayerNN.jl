

train_images_file_01 = "./MNIST/train-images-01"
train_labels_file_01 = "./MNIST/train-labels-01"
test_images_file_01 = "./MNIST/test-images-01"
test_labels_file_01 = "./MNIST/test-labels-01"

train_images_file_09 = "./MNIST/train-images-09"
train_labels_file_09 = "./MNIST/train-labels-09"
test_images_file_09 = "./MNIST/test-images-09"
test_labels_file_09 = "./MNIST/test-labels-09"

PIXEL_DEPTH = 255

function read_Int32(file::IOStream)
    number = read(file, Int32)
    return hton(number)
end

# read training data from image file
function read_image_file(filename::String)
    f = open(filename)
    magic_numner = read_Int32(f)
    num_imgs = read_Int32(f)
    num_rows = read_Int32(f)
    num_cols = read_Int32(f)
    image_size = num_rows * num_cols
    bin_data = read(f, num_imgs*image_size)
    data = Float64.(bin_data)
    data = data ./ PIXEL_DEPTH
    # 784 x N, when using, get one image from data by using data[:,1], you will get a 784x1 vector
    data = reshape(data, (image_size, num_imgs))
    close(f)
    return data
end

function read_label_file(filename::String)
    f = open(filename)
    magic_numner = read_Int32(f)
    num_imgs = read_Int32(f)
    bin_data= read(f, num_imgs)
    data = Int64.(bin_data)
    close(f)
    return data
end

function load_train_images(flag::Bool)
    if flag
        return read_image_file(train_images_file_01)
    else
        return read_image_file(train_images_file_09)
    end

end

function load_train_labels(flag::Bool)
    if flag
        return read_label_file(train_labels_file_01)
    else
        return read_label_file(train_labels_file_09)
    end
end

function load_test_images(flag::Bool)
    if flag
        return read_image_file(test_images_file_01)
    else
        return read_image_file(test_images_file_09)
    end
end

function load_test_labels(flag::Bool)
    if flag
        return read_label_file(test_labels_file_01)
    else
        return read_label_file(test_labels_file_09)
    end
end

glorot_normal(dims...) = randn(dims...) .* sqrt(2.0/sum(dims))

function initialize_weights_bias(num_input,num_hidden,num_output)
    W1 = glorot_normal(num_hidden, num_input)
    B1 = zeros(num_hidden, 1)
    W2 = glorot_normal(num_output, num_hidden)
    B2 = zeros(num_output, 1)
    dict_params = Dict(
        "W1" => W1,
        "B1" => B1,
        "W2" => W2,
        "B2" => B2)
    return dict_params
end

function forward_calculation(X,dict_params::Dict)
    W1=dict_params["W1"]
    B1=dict_params["B1"]
    W2=dict_params["W2"]
    B2=dict_params["B2"]
    # print W1,X,b1
    Z1= W1 * X .+ B1
    A1 = tanh.(Z1)
    Z2= W2 * A1 .+ B2
    A2 = sigmoid.(Z2)
    #assert(A2.shape == (1, X.shape[1]))
    cache = Dict(
            "Z1" => Z1,
            "A1" => A1,
            "Z2" => Z2,
            "A2" => A2)
    return A2, cache
end

function calculate_loss(A2::Array{Float64,2}, Y::Array{Float64,2})
    t=0.00000000001
    p1 = (log.(A2 .+ t)) .* Y
    p2 = (log.(1 .- A2 .+ t)) .* (1 .- Y)
    p3 = p1 .+ p2
    p4 = sum(p3, dims=1)
    cost = -1 * p4 / size(A2,1)
    return cost
end

function back_propagation(dict_params::Dict, dict_cache::Dict, X, Y)
    W1 = dict_params["W1"]
    W2 = dict_params["W2"]
    A1 = dict_cache["A1"]
    A2 = dict_cache["A2"]
    Z1 = dict_cache["Z1"]

    dZ2 = A2 - Y
    dW2 = dZ2 * transpose(A1)
    dB2 = sum(dZ2, dims=1)

    p1 = transpose(W2) * dZ2
    p2 = 1 .- (A1 .* A1)
    dZ1 = p1 .* p2
    dW1 = dZ1 * transpose(X)
    dB1 = sum(dZ1, dims=1)

    dict_grads = Dict("dW1" => dW1,
             "dB1" => dB1,
             "dW2" => dW2,
             "dB2" => dB2)
    return dict_grads
end

function update_para(dict_params::Dict, dict_grads::Dict, learning_rate::Float64)
    W1 = dict_params["W1"]
    B1 = dict_params["B1"]
    W2 = dict_params["W2"]
    B2 = dict_params["B2"]

    dW1 = dict_grads["dW1"]
    dB1 = dict_grads["dB1"]
    dW2 = dict_grads["dW2"]
    dB2 = dict_grads["dB2"]

    W1 = W1 .- learning_rate .* dW1
    B1 = B1 .- learning_rate .* dB1
    W2 = W2 .- learning_rate .* dW2
    B2 = B2 .- learning_rate .* dB2

    dict_params = Dict("W1" => W1,
                  "B1" => B1,
                  "W2" => W2,
                  "B2" => B2)
    return dict_params
end

function image2vector(image)
    v = reshape(image,(784,1))
    return v
end

function softmax(x)
    y = argmax(x)[1] - 1
    return y
end

function sigmoid(x)
    y = 1 / (1 + exp.(-1 .* x))
    return y
end

function Test(test_images, test_labels, dict_params::Dict)
    correct_count=0
    #test_loop = 10000
    test_loop = size(test_images, 2)
    print("test loop:$test_loop\n")
    for i= 1:test_loop
        img_test = test_images[:,i]
        #vector_image = image2vector(img_test)
        label_value = test_labels[i]
        aa2, xxx = forward_calculation(img_test, dict_params)
        predict_value = softmax(aa2)
        if predict_value == Int64(label_value)
            correct_count += 1
        end
    end
    return correct_count
end

# if flag == true, 01, else 09
function load_data(flag::Bool)
    train_images = load_train_images(flag)
    train_labels = load_train_labels(flag)
    test_images = load_test_images(flag)
    test_labels = load_test_labels(flag)
    return train_images, train_labels, test_images, test_labels
end

print("start, reading file...\n")
start_time = time()
print("$start_time\n")

train_images, train_labels, test_images, test_labels = load_data(false)

print("initializing...\n")

num_input = 28*28
# 32 for 0-9, 10 for 0-1
num_hidden = 32
# 2 for 0/1, if 0~9, then =10
num_output = 10

learning_rate = 0.001

dict_params = initialize_weights_bias(num_input, num_hidden, num_output)

print("training...\n")

# train
# loop = 50000
loop = size(train_images,2)
print("loop:$loop\n")
for i = 1:loop
    global dict_params
    #print("i:$i")
    label_value = train_labels[i]
    label_train = zeros(num_output, 1)
    # set [0,0] to [0,1] or [1,0] according to label_value's value
    label_train[Int64(label_value)+1] = 1
    # julia is 1-based array, so if label_value=0, then set to [1,0]
    # if lable_value = 1, then set to [0,1]

#    img_train = train_images[:, i]
#    img_vector = image2vector(img_train)
    img_vector = train_images[:, i]

    A2, dict_cache = forward_calculation(img_vector, dict_params)
    pre_label = softmax(A2)
    cost = calculate_loss(A2, label_train)
    dict_grads = back_propagation(dict_params, dict_cache, img_vector, label_train)
    dict_params = update_para(dict_params, dict_grads, learning_rate)

    dict_grads["dW1"] .= 0
    dict_grads["dW2"] .= 0
    dict_grads["dB1"] .= 0
    dict_grads["dB2"] .= 0
end

# testing
test_count = size(test_images,2)
print("testing... test count:$test_count\n")
correct_count = Test(test_images, test_labels, dict_params)
correct_percent = correct_count / test_count
print("correct count: $correct_count, percent=$correct_percent\n")
    #Test(test_images, test_labels, loadParam)

end_time = time()
print("$end_time\n")
print(end_time - start_time)

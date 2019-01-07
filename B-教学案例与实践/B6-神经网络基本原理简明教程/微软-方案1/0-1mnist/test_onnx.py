import onnx
import numpy as np
from onnx_tf.backend import prepare
from test_origin import Testmodel, TestmodelFC, Cmodel
import tensorflow as tf


def conv_test():
    conv1_weights = tf.Variable(np.load("conv1w.npy").astype(np.float32))
    conv1_biases = tf.Variable(np.load("conv1b.npy").astype(np.float32))
    d = tf.placeholder(tf.float32, shape=[1, 28, 28, 1])
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    conv = tf.nn.conv2d(d,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    tf_model = tf.nn.bias_add(conv, conv1_biases)
    

    model = onnx.load("test_conv.onnx")
    tf_rep = prepare(model)
    npmodel = Testmodel(1)
    img = np.load("img.npy")
    for ele in img:
        data = ele.reshape([1, 28, 28, 1])
        tf_result = sess.run(tf_model, feed_dict={d:data})
        tf_result = np.array(tf_result)
        npresult = npmodel.infer(data)
        data = ele.reshape([1, 1, 28, 28])
        output = np.array(tf_rep.run(data))
        output = output[0]
        print(output.shape)
        output = np.swapaxes(output, 2, 3)
        output = np.swapaxes(output, 1, 3)

        print(output[0][0][1])
        print(tf_result[0][0][1])
        print(npresult[0][0][1])

        break

def fc_test():
    model = onnx.load("test_fc.onnx")
    tf_rep = prepare(model)
    npmodel = TestmodelFC(1)
    img = np.load("img.npy")
    for ele in img:
        data = ele.reshape([1, 784])
        npresult = npmodel.infer(data)
        output = tf_rep.run(data)
        print(output[0])
        print(npresult[0])
        break

def model_test():
    model = onnx.load("model.onnx")
    onnx.checker.check_model(model)
    tf_rep = prepare(model)
    img = np.load("img.npy")
    for ele in img:
        data = ele.reshape([1, 28, 28, 1])
        data = np.swapaxes(data, 1, 3)
        data = np.swapaxes(data, 2, 3)
        output = np.array(tf_rep.run(data))
        output = output.reshape((1, 2))
        print(output)




if __name__ == "__main__":
    model_test()
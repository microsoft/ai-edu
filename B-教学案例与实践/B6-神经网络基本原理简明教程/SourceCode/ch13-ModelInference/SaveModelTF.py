import tensorflow as tf
SEED = 46
def main():
  # first generate weights and bias used in conv layers
  conv1_weights = tf.Variable(
    tf.truncated_normal([5, 5, 3, 8],  # 5x5 filter, depth 8.
              stddev=0.1,
              seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([8]))

  # data and out is placeholder used for input and output data in practice
  data = tf.placeholder(dtype=tf.float32, name="data", shape=[8, 32, 32, 3])
  out = tf.placeholder(dtype=tf.float32, name="out", shape=[8, 32, 32, 8])

  # as the structure of the simple model
  def model():
    conv = tf.nn.conv2d(data,
            conv1_weights,
            strides=[1, 1, 1, 1],
            padding='SAME', name="conv")
    out = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases), name="relu")

  # saver is used for saving model
  saver = tf.train.Saver()
  with tf.Session() as sess:
    model()
    # initialize all variables
    tf.global_variables_initializer().run()
    # save the model in the file path
    saver.save(sess, './model')

if __name__ == "__main__":
  main()
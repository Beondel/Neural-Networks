import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hl1_nodes = 500
hl2_nodes = 500
hl3_nodes = 500
n_classes = 10
batch_size = 100

x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
y = tf.placeholder(tf.float32)


def deep_nn(input):
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([28 * 28, hl1_nodes], mean=0.0, stddev=0.5)),
                      'biases': tf.Variable(tf.random_normal([hl1_nodes], mean=0.0, stddev=0.5))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([hl1_nodes, hl2_nodes], mean=0.0, stddev=0.5)),
                      'biases': tf.Variable(tf.random_normal([hl2_nodes], mean=0.0, stddev=0.5))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([hl2_nodes, hl3_nodes], mean=0.0, stddev=0.5)),
                      'biases': tf.Variable(tf.random_normal([hl3_nodes], mean=0.0, stddev=0.5))}

    output_layer = {'weights': tf.Variable(tf.random_normal([hl3_nodes, n_classes], mean=0.0, stddev=0.5)),
                    'biases': tf.Variable(tf.random_normal([n_classes], mean=0.0, stddev=0.5))}

    l1 = tf.nn.relu_layer(input, hidden_layer_1['weights'], hidden_layer_1['biases'])
    l2 = tf.nn.relu_layer(l1, hidden_layer_2['weights'], hidden_layer_2['biases'])
    l3 = tf.nn.relu_layer(l2, hidden_layer_3['weights'], hidden_layer_3['biases'])
    return tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])


def train(x):
    prediction = deep_nn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    n_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples / batch_size)):
                mnist_x, mnist_y = mnist.train.next_batch(batch_size)
                i, c = sess.run([optimizer, cost], feed_dict={x: mnist_x, y: mnist_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train(x)

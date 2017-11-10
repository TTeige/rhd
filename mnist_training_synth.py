import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import time
import os
from datetime import timedelta

h5f = h5py.File('data/MNIST_synthetic.h5', 'r')

X_train = h5f['train_dataset'][:]
y_train = h5f['train_labels'][:]
X_val = h5f['valid_dataset'][:]
y_val = h5f['valid_labels'][:]
X_test = h5f['test_dataset'][:]
y_test = h5f['test_labels'][:]

print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)

h5f.close()


def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    # Initialize figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2 * nrows))

    for i, ax in enumerate(axes.flat):

        # Pretty string with actual number
        true_number = ''.join(str(x) for x in cls_true[i] if x != 10)

        if cls_pred is None:
            title = "True: {0}".format(true_number)
        else:
            # Pretty string with predicted number
            pred_number = ''.join(str(x) for x in cls_pred[i] if x != 10)
            title = "True: {0}, Pred: {1}".format(true_number, pred_number)

        ax.imshow(images[i, :, :, 0], cmap='binary')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])


def conv_weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())


def fc_weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    return tf.Variable(tf.constant(1.0, shape=shape))


def conv_layer(input,  # The previous layer.
               n_channels,  # Num. channels in prev. layer.
               f_size,  # Width and height of each filter.
               n_filters,  # Number of filters.
               weight_name,  # Name of variable containing the weights
               pooling=True):  # Use 2x2 max-pooling.

    # Create weights and biases
    weights = conv_weight_variable(weight_name, [f_size, f_size, n_channels, n_filters])
    biases = bias_variable([n_filters])

    layer = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
    layer = tf.nn.relu(layer + biases)

    # Use pooling to down-sample the image resolution?
    if pooling:
        layer = tf.nn.max_pool(layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')

    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    layer_flat = tf.reshape(layer, [-1, num_features])

    # Return the flattened layer and the number of features.
    return layer_flat, num_features


def fc_layer(input,  # The previous layer.
             num_inputs,  # Num. inputs from prev. layer.
             num_outputs,  # Num. outputs.
             weight_name,  # Name of variable containing the weights
             relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = fc_weight_variable(weight_name, shape=[num_inputs, num_outputs])
    biases = bias_variable([num_outputs])

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if relu:
        layer = tf.nn.relu(layer)

    return layer


def softmax_function(input,  # Previous layer
                     num_inputs,  # Number of inputs from previous layer
                     num_outputs,  # Number of outputs
                     weight_name):  # Name of variable containing the weights

    # Create weights and biases
    weights = fc_weight_variable(weight_name, [num_inputs, num_outputs])
    biases = bias_variable([num_outputs])

    # Softmax
    logits = tf.matmul(input, weights) + biases

    return logits, weights


# We processed image size to be 64
img_size = 64

# Number of channels: 1 because greyscale
num_channels = 1

# Number of digits
num_digits = 3

# Number of output labels
num_labels = 11

# Convolutional Layer 1
filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
num_filters1 = 16  # There are 16 of these filters.

# Convolutional Layer 2
filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
num_filters2 = 32  # There are 36 of these filters.

# Convolutional Layer 3
filter_size3 = 5  # Convolution filters are 5 x 5 pixels.
num_filters3 = 64  # There are 48 of these filters.

# Fully-connected layer
fc_size = 64  # Number of neurons in fully-connected layer.

# Images placeholder
x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, num_channels), name='x')

# Labels placeholder
y_true = tf.placeholder(tf.int64, shape=[None, num_digits], name='y_true')

keep_prob = tf.placeholder(tf.float32)

conv_1, w_c1 = conv_layer(x, num_channels, filter_size1, num_filters1, 'w_c1', True)
conv_2, w_c2 = conv_layer(conv_1, num_filters1, filter_size2, num_filters2, 'w_c2', True)
conv_3, w_c3 = conv_layer(conv_2, num_filters2, filter_size3, num_filters3, 'w_c3', False)
dropout = tf.nn.dropout(conv_3, keep_prob)
flatten, num_features = flatten_layer(dropout)
fc_1 = fc_layer(flatten, num_features, fc_size, 'w_fc1', relu=True)
logits_1, w_s1 = softmax_function(fc_1, fc_size, num_labels, 'w_s1')
logits_2, w_s2 = softmax_function(fc_1, fc_size, num_labels, 'w_s2')
logits_3, w_s3 = softmax_function(fc_1, fc_size, num_labels, 'w_s3')
# logits_4, w_s4 = softmax_function(fc_1, fc_size, num_labels, 'w_s4')
# logits_5, w_s5 = softmax_function(fc_1, fc_size, num_labels, 'w_s5')

# y_pred = [logits_1, logits_2, logits_3, logits_4, logits_5]
y_pred = [logits_1, logits_2, logits_3]

# The class-number is the index of the largest element.
y_pred_cls = tf.transpose(tf.argmax(y_pred, dimension=2))
loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=y_true[:, 0]))
loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_2, labels=y_true[:, 1]))
loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3, labels=y_true[:, 2]))
# loss4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_4, labels=y_true[:, 3]))
# loss5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_5, labels=y_true[:, 4]))

# loss = loss1 + loss2 + loss3 + loss4 + loss5
loss = loss1 + loss2 + loss3

# We use global_step as a counter variable
global_step = tf.Variable(0)

# The learning rate is initially set to 0.05
start_learning_rate = 0.1

# Apply exponential decay to the learning rate
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 10000, 0.96)

# Use the Adagrad optimizer
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)


def accuracy(predictions, labels):
    return (100.0 * np.sum(predictions == labels) / predictions.shape[1] / predictions.shape[0])


session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
session.run(tf.global_variables_initializer())

# Batch size
batch_size = 64

# Number of steps between each update
display_step = 5000

# Dropout
dropout = 0.5
saver = tf.train.Saver()

save_dir = 'checkpoints/'

# Create directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'mnist_net2')
saver.restore(sess=session, save_path=save_path)
print("Model restored")
total_iterations = 0


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for step in range(num_iterations):

        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :, :, :]
        batch_labels = y_train[offset:(offset + batch_size), :]

        feed_dict_train = {x: batch_data, y_true: batch_labels, keep_prob: dropout}

        # Run the optimizer using this batch of training data.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every x iterations.
        if step % display_step == 0:
            # Calculate the accuracy on the training-set.
            batch_predictions = session.run(y_pred_cls, feed_dict=feed_dict_train)
            print("Minibatch accuracy at step %d: %.4f" % (step, accuracy(batch_predictions, batch_labels)))

            # Calculate the accuracy on the validation-set
            val_predictions = session.run(y_pred_cls, {x: X_val, y_true: y_val, keep_prob: 1.})
            print("Validation accuracy: %.4f" % accuracy(val_predictions, y_val))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Difference between start and end-times.
    time_dif = time.time() - start_time

    # Calculate the accuracy on the test-set
    test_predictions = session.run(y_pred_cls, {x: X_test, y_true: y_test, keep_prob: 1.})

    print("Test accuracy: %.4f" % accuracy(test_predictions, y_test))
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    saver.save(sess=session, save_path=save_path)
    print('Model saved in file: {}'.format(save_path))


optimize(num_iterations=1)
test_pred = session.run(y_pred_cls, feed_dict={x: X_test, y_true: y_test, keep_prob: 1.0})

session.close()
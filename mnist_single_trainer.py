import tensorflow as tf
import h5py
import argparse
import numpy as np
import cv2


def convert_img(img):
    img_data = cv2.bitwise_not(img)
    # img_data = np.array(img_data, dtype=np.float32)
    # img_data = img_data.flatten()
    # img_data = [float(x) * 1.0 / 255.0 for x in img_data]
    # img_data = np.reshape(img_data, [64, 64]).astype(np.float32)

    gray_channel = cv2.resize(img_data, (84, 28), interpolation=cv2.INTER_CUBIC)

    reshaped = np.zeros((128, 128))
    p = np.array(gray_channel)
    y_off = 50
    x_off = 22
    reshaped[y_off:p.shape[0] + y_off, x_off:p.shape[1] + x_off] = p
    # reshaped = cv2.blur(reshaped, (1,1))
    img_data = [reshaped]
    # img_data = np.expand_dims(img_data, axis=3)
    # img_data = img_data.tolist()
    return img_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def convDeepnn(x):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 128, 128, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([5, 6, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_fc1 = weight_variable([16 * 16 * 128, 2048])
    b_fc1 = bias_variable([2048])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 16 * 16 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([2048, 1000])
    b_fc2 = bias_variable([1000])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


def run(args):
    x = tf.placeholder(tf.float32, [None, 128, 128], name='x')
    y_ = tf.placeholder(tf.int64, [None, 1000], name='y_')
    y_conv, keep_prob = convDeepnn(x)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='cor_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    predictor = tf.nn.softmax(y_conv, 1, name='predictor')
    predictor2 = tf.argmax(y_conv, 1)
    saver = tf.train.Saver()

    h5f = h5py.File('data/MNIST_synthetic.h5', 'r')

    X_train = h5f['train_dataset'][:]
    y_train = h5f['train_labels'][:]
    X_val = h5f['valid_dataset'][:]
    y_val = h5f['valid_labels'][:]
    X_test = h5f['test_dataset'][:]
    y_test = h5f['test_labels'][:]

    if args.train:

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        batch_size = 50

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('model/'))
            for i in range(200000):

                offset = (i * batch_size) % (y_train.shape[0] - batch_size)
                batch_data = X_train[offset:(offset + batch_size), :]
                batch_labels = y_train[offset:(offset + batch_size), :]

                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch_data, y_: batch_labels, keep_prob: 1.0})
                    print('step {}, training accuracy {}'.format(i, train_accuracy))
                train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})
            saver.save(sess, 'model/mnist_model_2')

            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: X_val[:1000], y_: y_val[:1000], keep_prob: 1.0}))

    else:
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('model/'))
            img = convert_img(cv2.imread("/mnt/remote/extracted_fields/fs10061402175155/10_27fs10061402175155.jpg", 0))
            feed_dict = {x: img, keep_prob: 1.0}
            # feed_dict = {x: img, y_: np.zeros([1, 1000], dtype=np.int64), keep_prob: 1.0}
            print(sess.run(predictor, feed_dict))
            # print(sess.run(correct_prediction, feed_dict))
            print(sess.run(predictor2, feed_dict))
            # print(sess.run(predictor, feed_dict={x: [X_test[0]], keep_prob: 1.0}))
            # cv2.imshow("1", X_test[0])
            # cv2.imshow("", img[0])
            # cv2.waitKey()


def main():
    arg_parser = argparse.ArgumentParser(
        description="Predict images at the given path. Model based on a synthetic mnist dataset")
    arg_parser.add_argument("--images", "-i", type=str,
                            help="path to root of directory structure that is to be predicted")
    arg_parser.add_argument("--output", "-o", type=str, help="Output path of csv file containing the predictions")
    arg_parser.add_argument("--model", "-m", type=str, help="Path to the model that is to be restored", default="")
    arg_parser.add_argument("--meta_graph", "-M", type=str, help="Path to the metagraph for restoration of the model",
                            default="")
    arg_parser.add_argument("--checkpoint", "-c", type=str, help="Restore from checkpoint if path is provided",
                            default="")
    arg_parser.add_argument("--train", "-t", action="store_true", help="set the program to train on a given dataset")
    args = arg_parser.parse_args()
    print(args)
    run(args)


if __name__ == '__main__':
    main()

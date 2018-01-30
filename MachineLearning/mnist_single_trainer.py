import tensorflow as tf
import h5py
import argparse
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data


class Trainer:
    def __init__(self):

        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y_ = tf.placeholder(tf.int64, [None, 10], name='y_')

        self.batch_size = 50
        self.predictor = None
        self.predictor2 = None
        self.y_conv = None
        self.keep_prob = 1.0

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def convDeepnn(self, x):
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return y_conv, keep_prob

    def run_training(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        y_conv, keep_prob = self.convDeepnn(self.x)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1), name='cor_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for i in range(200000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    self.x: batch[0], self.y_: batch[1], keep_prob: self.keep_prob})

                print('step {}, training accuracy {}'.format(i, train_accuracy))

            train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            self.x: mnist.test.images, self.y_: mnist.test.labels, keep_prob: 1.0}))

    def initialize_tensors(self, y_conv):
        self.predictor = tf.nn.softmax(y_conv, 1, name='predictor')
        self.predictor2 = tf.argmax(y_conv, 1)


# Use this to normalize and flatten the image
def convert_img(img):
    # img_data = cv2.bitwise_not(img)
    # img_data = np.array(img_data, dtype=np.float32)
    # img_data = img_data.flatten()
    # img_data = [float(x) * 1.0 / 255.0 for x in img_data]
    # img_data = np.reshape(img_data, [64, 64]).astype(np.float32)

    # gray_channel = cv2.resize(img_data, (28, 28), interpolation=cv2.INTER_CUBIC)
    #
    # reshaped = np.zeros((28, 28))
    # p = np.array(gray_channel)
    # y_off = 50
    # x_off = 22
    # reshaped[y_off:p.shape[0] + y_off, x_off:p.shape[1] + x_off] = p
    # reshaped = cv2.blur(reshaped, (1,1))
    # img_data = [reshaped]
    # img_data = np.expand_dims(img_data, axis=3)
    # img_data = img_data.tolist()
    return img


def run(args):
    trainer = Trainer()
    # saver = tf.train.Saver()

    if args.train:

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if tf.train.latest_checkpoint("model/"):
                saver.restore(sess, tf.train.latest_checkpoint('model/'))

            trainer.run_training()
            saver.save(sess, 'model/mnist_model_2')

    else:
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('model/'))
            img = convert_img(cv2.imread("/mnt/remote/extracted_fields/fs10061402175155/10_27fs10061402175155.jpg", 0))

            y_conv, keep_prob = trainer.convDeepnn(trainer.x)
            trainer.initialize_tensors(y_conv)


            feed_dict = {trainer.x: img, keep_prob: 1.0}
            # feed_dict = {x: img, y_: np.zeros([1, 1000], dtype=np.int64), keep_prob: 1.0}
            print(sess.run(trainer.predictor, feed_dict))
            # print(sess.run(correct_prediction, feed_dict))
            print(sess.run(trainer.predictor2, feed_dict))
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

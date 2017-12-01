import tensorflow as tf
import sys
import cv2
import numpy as np
import csv
import os


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

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob

def convert_img(img):
    img_data = np.array(img)
    img_data = img_data.flatten()
    img_data = [float(x)*1.0/255.0 for x in img_data]
    return img_data

def run(model_path, img_path):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_conv, keep_prob = convDeepnn(x)
    y_conv = tf.nn.softmax(y_conv)
    saver = tf.train.Saver()
    print(img_path)
    with open('predictions.csv', 'w') as csv_file:
        fieldnames = ['filename', 'prediction', 'confidence']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            prediction = tf.argmax(y_conv, 1)
            for dir, dirs, fns in os.walk(img_path):
                for fn in fns:
                    print(dir+"/"+fn)
                    img_data = convert_img(cv2.imread(dir+"/"+fn, 0))
                    pred = prediction.eval(feed_dict={x: [img_data], keep_prob:1.0}, session=sess)
                    prob = y_conv.eval(feed_dict={x: [img_data], keep_prob:1.0}, session=sess)
                    writer.writerow({'filename':dir+"/"+fn, 'prediction':pred, 'confidence':prob})

def main(argv):
    if len(argv) < 3:
        print("python3 predictor.py <model_name> <PATH/to/images>")
        return
    run(argv[1], argv[2])

if __name__ == '__main__':
    main(sys.argv)

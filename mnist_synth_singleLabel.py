import random
import h5py
import numpy as np
import cv2
from sklearn import datasets
from sklearn.utils import shuffle
from scipy.misc import imresize
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


def plot_image(img):
    """Helper function for displaying a single image
    """
    plt.imshow(img, interpolation='nearest')


def plot_images(img, labels, nrows, ncols):
    """Helper function used to display digits
    """
    plt.figure(figsize=(min(16, ncols * 2), min(16, nrows * 2)))
    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)
        # Reshape every image to a square array 2d array
        if img[i].shape == (64, 64, 1):
            plt.imshow(img[i, :, :, 0], interpolation='nearest')
        else:
            plt.imshow(img[i], interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.title(labels[i])


def save_dataset(fp, i_train, l_train, i_test, l_test, i_val, l_val):
    h5f = h5py.File(fp, 'w')

    h5f.create_dataset('train_dataset', data=i_train)
    h5f.create_dataset('train_labels', data=l_train)
    h5f.create_dataset('test_dataset', data=i_test)
    h5f.create_dataset('test_labels', data=l_test)
    h5f.create_dataset('valid_dataset', data=i_val)
    h5f.create_dataset('valid_labels', data=l_val)

    h5f.close()


def fetch_mnist():
    # Fetch the MNIST dataset from mldata.org
    mnist = datasets.fetch_mldata('MNIST original')

    # Extract the data and labels
    X, y = mnist.data.astype(float), mnist.target.astype(int)

    # Reshape the original dataset which is flat to 28x28 images
    X = X.reshape(len(X), 28, 28)

    X, y = shuffle(X, y)

    return X, y


def create_dataset(width, height, max_digits, scale, new_height, new_width, dataset_size):
    images, labels = fetch_mnist()
    new_images, new_labels = generate_digit_sequences(images, labels, dataset_size,
                                                      height, width, max_digits, scale, new_width, new_height)

    boxes = [0 for _ in range(1000)]
    for label_range in new_labels:
        for i, label in enumerate(label_range):
            if label != 0:
                boxes[i] += 1
    print("Sequence distribution count {}".format(boxes))

    image_train, image_test, labels_train, labels_test = train_test_split(new_images, new_labels)
    idx = np.random.choice(range(0, len(image_test)), size=int(len(image_test) * 0.20), replace=False)
    image_val, labels_val = image_test[idx], labels_test[idx]
    image_test = np.delete(image_test, idx, axis=0)
    labels_test = np.delete(labels_test, idx, axis=0)
    save_dataset("data/MNIST_synthetic.h5", image_train, labels_train, image_test, labels_test, image_val, labels_val)


def concat_labels(labels, num_images):
    """ Concatenates a set of set of labels into a single numpy array
    """
    new_label = ""
    for i in range(num_images):
        if i < len(labels):
            new_label += str(labels[i])

    return int(new_label)


def generate_digit_sequences(data, labels, n, height, width, max_digits, scale, new_width, new_height):
    """ Helper function for automatically generating a new dataset of digit sequences
    """
    # Initialize numpy arrays
    images = np.zeros(shape=(n, height, width), dtype=np.uint8)
    generated_labels = np.zeros(shape=(n, 1000), dtype=np.int)

    # Number of training examples of each sequence length
    n_samples = n / max_digits

    # For every possible digit sequence length
    for i in range(1, max_digits + 1):

        # Pick n_samples images
        n_samples = int(n_samples)
        for j in range((i - 1) * n_samples, i * n_samples):
            if random.random() < 0.05:
                images[j] = np.ones(shape=(height, width), dtype=np.uint8)
                generated_labels[j][1000-1] = 1
            else:
                # Select i random digits from the original dataset
                selection = random.sample(range(0, len(data)), i)

                # Concatenate the digits and labels from
                images[j] = concat_images(data[selection], width, height, scale, new_width, new_height)
                generated_labels[j][concat_labels(labels[selection], max_digits)] = 1

    # Return the new dataset
    return images, generated_labels


def concat_images(images, width, height, scale, new_height, new_width):
    """ Scales, concatenates and centers a sequence of digits in an image
    """
    # Keep this
    num_digits = len(images)

    # Initialize a numpy array for the new image
    new_image = np.zeros(shape=(height, width), dtype="uint8")

    # Calculate the horizontal and vertical padding
    y_pad = (height - new_height) / 2
    x_pad = (width - num_digits * new_width) / 2

    # For every image passed to the function
    for i in range(num_digits):
        # Scale down the original image
        scaled = cv2.resize(images[i], (28, 28), interpolation=cv2.INTER_CUBIC)

        # Calculate the starting position
        x_offset = x_pad + (i * new_width)

        # Add the scaled image to the new image
        new_image[int(y_pad):int(height - y_pad), int(x_offset):int(x_offset + new_width)] = scaled

    # Return the newly centered image
    return new_image


def run():

    height, width = 128, 128
    max_digits = 3
    scale = 1
    new_height, new_width = 28, 28
    dataset_size = 400000

    create_dataset(width, height, max_digits, scale, new_height, new_width, dataset_size)


if __name__ == '__main__':
    run()

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, preprocessing

# Fetch the MNIST dataset from mldata.org
mnist = datasets.fetch_mldata('MNIST original')

# Extract the data and labels
X, y = mnist.data.astype(float), mnist.target.astype(int)

print("Original Shape")
print(X.shape, y.shape)

# Reshape the original dataset which is flat to 28x28 images
X = X.reshape(len(X), 28, 28)

print("\nNew Shape")
print(X.shape)


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


from sklearn.utils import shuffle

# Shuffle the dataset
X, y = shuffle(X, y)
# Height and width of our new synthethic images
height, width = 64, 64

# Maximum number of digits
max_digits = 3

# Scale the origina images down to 45% (12x12 pixels)
scale = 0.45

# New height and width of the scaled images
new_height, new_width = 12, 12

# We create a new dataset with 50,000 images
dataset_size = 50000

from scipy.misc import imresize


def concat_images(images):
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
        scaled = imresize(images[i], scale)

        # Calculate the starting position
        x_offset = x_pad + (i * new_width)

        # Add the scaled image to the new image
        new_image[int(y_pad):int(height - y_pad), int(x_offset):int(x_offset + new_width)] = scaled

    # Return the newly centered image
    return new_image

def concat_labels(labels, num_images=3):
    """ Concatenates a set of set of labels into a single numpy array
    """
    new_label = np.zeros(num_images, dtype=int)
    for i in range(num_images):
        if i < len(labels):
            new_label[i] = labels[i]
        else:
            new_label[i] = 10
    return new_label

import random


def generate_digit_sequences(data, labels, n):
    """ Helper function for automatically generating a new dataset of digit sequences
    """
    # Initialize numpy arrays
    X = np.zeros(shape=(n, height, width), dtype='uint8')
    y = np.zeros(shape=(n, max_digits), dtype=np.int32)

    # Number of training examples of each sequence length
    n_samples = n / max_digits

    # For every possible digit sequence length
    for i in range(1, max_digits + 1):

        # Pick n_samples images
        n_samples = int(n_samples)
        for j in range((i - 1) * n_samples, i * n_samples):
            if random.random() < 0.05:
                X[j] = np.ones(shape=(height, width), dtype="uint8")
                y[j] = np.full(max_digits, 10)
            else:
                # Select i random digits from the original dataset
                selection = random.sample(range(0, len(data)), i)

                # Concatenate the digits and labels from
                X[j] = concat_images(data[selection])
                y[j] = concat_labels(labels[selection])

    # Add an additional dimension to the image array
    X = np.expand_dims(X, axis=3)

    # Return the new dataset
    return X, y


# Generate a synthetic dataset of digit sequences with 50,000 new images
X_new, y_new = generate_digit_sequences(X, y, dataset_size)

print("Images", X_new.shape)
print("Labels", y_new.shape)

# Plot a histogram showing the class distribution
plt.subplot2grid((1, 2), (0, 0))
plt.hist(y_new.flatten())
plt.title("Class Distribution (10 = empty)")

# Plot a histogram showing the sequence length distribution
plt.subplot2grid((1, 2), (0, 1))
plt.hist((y_new != 10).sum(1), color='r', bins=4)
plt.xlim(0, 3)
plt.title("Sequence Length Distribution")
plt.show()

from sklearn.cross_validation import train_test_split

# Create a training and test set
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new)

# Select some random images to be used in our validation set
idx = np.random.choice(range(0, len(X_test)), size=int(len(X_test) * 0.20), replace=False)

# Construct our validation set and remove the images from the test set
X_val, y_val = X_test[idx], y_test[idx]

# Remove the validation samples from the testset
X_test = np.delete(X_test, idx, axis=0)
y_test = np.delete(y_test, idx, axis=0)

print("Training", X_train.shape, y_train.shape)
print("Test", X_test.shape, y_test.shape)
print('Validation', X_val.shape, y_val.shape)

import h5py

# Create file
h5f = h5py.File('data/MNIST_synthetic.h5', 'w')

# Store the datasets
h5f.create_dataset('train_dataset', data=X_train)
h5f.create_dataset('train_labels', data=y_train)
h5f.create_dataset('test_dataset', data=X_test)
h5f.create_dataset('test_labels', data=y_test)
h5f.create_dataset('valid_dataset', data=X_val)
h5f.create_dataset('valid_labels', data=y_val)

# Close the file
h5f.close()

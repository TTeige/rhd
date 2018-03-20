import cv2
import numpy as np
import keras
from random import shuffle


class RHData(keras.utils.Sequence):
    def __init__(self, data, batch_size):
        """
        Class which encapsulates the data needed for training the classification model
        :param data: an array of tuples containing the image file path and the label for that image (path, label)
        """
        if data is None:
            raise TypeError("Input data is None.")
        elif len(data) < 5000:
            raise ValueError(
                "Input data is to small. Data size is {}. At least 5000 samples is required".format(len(data)))

        self.x = []
        self.y = []

        for val in data:
            self.x.append(val[0])
            self.y.append(val[1])

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, item):
        batch_x = self.x[item * self.batch_size:(item + 1) * self.batch_size]
        batch_y = self.y[item * self.batch_size:(item + 1) * self.batch_size]

        x = []
        for filename in batch_x:
            img = cv2.imread(filename, 0)
            img_flat = img.flatten()
            img_flat = [v / 255 for v in img_flat]
            x.append(np.array(img_flat))

        return np.array([x, batch_y])

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import math
from scipy.ndimage import affine_transform
from scipy.signal import argrelmin
import os


class GaussianNormalDistributionCluster:
    """
    GaussianNormalDistributionCluster provides methods for extracting the density distribution of an image, it's summed gaussian normal distributions and it's minimas for digit seperation.
    In order to render the plots, matplotlib.pyplot.show() must be called after the rendering methods are called.
    The load_image(path) method must be called before using any other method.
    """

    def __init__(self, num_components=3):
        """
        :param num_components: number of gaussian normal distributions
        """
        self.image = None
        self.components = num_components

    @staticmethod
    def gaussian(x, mu, sig, weight):
        """
        Creates a gaussian normal distribution
        :param x: ndarray of points along the x-axis
        :param mu: standard deviation
        :param sig: covariance
        :param weight: the weight for the normal distribution
        :return: a ndarray containing the points for the normal distribution
        """
        return (np.exp(-np.power(x - mu, 2.) / (2 * sig)) / (math.sqrt(2 * math.pi) * math.sqrt(sig))) * weight

    def load_image(self, path):
        """
        Loads an image in grayscale using opencv
        :param path: path to the image
        :return: ndarray of pixel values, grayscale
        """
        self.image = cv2.imread(path, 0)
        if self.image is None:
            print("Unable to load image, check path")
            raise ValueError
        return self.image

    def get_x_density(self):
        """
        Creates a 1d array containing the location of pixel values on the x-axis above a threshold, load_image must be called first
        :return: list of pixel locations
        """
        if self.image is None:
            raise ValueError
        w, h = self.image.shape

        np.random.seed(0)
        affine = np.array([[1, 0, 0], [-0.1, 1, 0], [0, 0, 1]])
        img = affine_transform(self.image, affine, cval=255)
        img_flat = img.flatten()
        img_flat = [v / 255 for v in img_flat]

        x_density = []
        for i in range(0, len(img_flat)):
            if img_flat[i] < 0.1:
                x_density.append(np.array([i % w]))

        return np.array(x_density)

    def find_minimas(self, summed_gaussian=None):
        """
        Returns local minimas of the gaussian function
        :param summed_gaussian: sum of gaussian normal distributions. If None, the method will retrieve a summed gaussian for the given number of components
        :return: local minimas
        """
        if summed_gaussian is None:
            summed_gaussian = self.get_summed_gaussian()
        minims = argrelmin(summed_gaussian)
        return minims

    @staticmethod
    def render_hist(x_density, num_bins=128):
        """
        Render method for a histogram
        :param x_density: list of x-axis pixel locations
        :param num_bins: number of bins to separate the values in to
        :return:
        """
        plt.hist(x_density, histtype='bar', normed=True, bins=num_bins)

    @staticmethod
    def render_dist(gaussian):
        """
        Render the given gaussian distribution
        :param gaussian: list containing the gaussian distribution
        :return:
        """
        plt.plot(gaussian)

    def get_summed_gaussian(self, x_density=None, init_weight=1 / 3):
        """
        Creates and summarizes the gaussian normal distributions
        :param x_density: list of pixel locations on the x-axis
        :param init_weight: initial weight for the distributions
        :return: summed gaussian distribution
        """

        if x_density is None:
            x_density = self.get_x_density()
        weights = np.full(self.components, init_weight)
        gmm = GaussianMixture(n_components=self.components, weights_init=weights)
        gmm.fit(x_density)

        mu = gmm.means_.flatten()
        sig = gmm.covariances_.flatten()
        combined = []
        for i in range(0, len(mu)):
            combined.append([mu[i], sig[i], gmm.weights_[i]])
        gausses = []
        for v in combined:
            g = self.gaussian(np.arange(self.image.shape[0]), v[0], v[1], v[2])
            gausses.append(g)
        gausses = np.array(gausses)
        sum_g = gausses.sum(axis=0)

        return sum_g

# def run():
#     np.random.seed(0)
#     gnc = GaussianNormalDistributionCluster()
#     img_name = "8_27fs10061408024922.jpg"
#     folder = "/mnt/remote/Yrke/spesifikke_felt/555/"
#     full_name = os.path.join(folder, img_name)
#     img = gnc.load_image(full_name)
#     x_density = gnc.get_x_density()
#     gnc.render_hist(x_density)
#     sum_g = gnc.get_summed_gaussian(x_density)
#     gnc.render_dist(sum_g)
#     mins = gnc.find_minimas(sum_g)
#     cv2.line(img, (mins[0][0], img.shape[1]), (mins[0][0], 0), (0, 0, 0))
#     cv2.line(img, (mins[0][1], img.shape[1]), (mins[0][1], 0), (0, 0, 0))
#     plt.show()
#
#
#
# if __name__ == '__main__':
#     run()

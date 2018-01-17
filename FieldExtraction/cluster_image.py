import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import math
from scipy.ndimage import affine_transform
from scipy.signal import argrelmin


class GaussianNormalDistributionCluster:
    def __init__(self):
        self.image = None

    @staticmethod
    def gaussian(x, mu, sig, weight):
        return (np.exp(-np.power(x - mu, 2.) / (2 * sig)) / (math.sqrt(2 * math.pi) * math.sqrt(sig))) * weight

    def load_image(self, path):
        self.image = cv2.imread(path, 0)
        if self.image is None:
            print("Unable to load image, check path")
            raise ValueError

    def get_x_density(self):
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

    def find_minimas(self, init_weight=1/3):
        sum_g = self.get_summed_gaussian(init_weight)
        minims = argrelmin(sum_g)
        return minims

    def get_hist(self, x_density, num_bins=128, render=False):
        n, bins, patches = plt.hist(x_density, histtype='bar', normed=True, bins=num_bins)
        if render:
            plt.plot(bins)
        return n, bins, patches

    def render_dist(self, gaussian):
        plt.plot(gaussian)

    def get_summed_gaussian(self, init_weight):
        x_density = self.get_x_density()
        gmm = GaussianMixture(n_components=3, weights_init=[init_weight, init_weight, init_weight])
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

        sum_g = gausses[0] + gausses[1] + gausses[2]

        return sum_g

# img_name = "8_27fs10061408024922.jpg"
# folder = "/mnt/remote/Yrke/spesifikke_felt/555/"
# full_name = os.path.join(folder, img_name)
# img = cv2.imread(full_name, 0)
# img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
# w, h = img.shape

# affine = np.array([[1, 0, 0], [-0.1, 1, 0], [0, 0, 1]])
# img = affine_transform(img, affine, cval=255)
# img_flat = img.flatten()
# # img_flat = [print(v) if v == None else v for v in img_flat]
# img_flat = [v / 255 for v in img_flat]
#
# x_density = []
# for i in range(0, len(img_flat)):
#     if img_flat[i] < 0.1:
#         x_density.append(np.array([i % w]))
#
# x_density = np.array(x_density)
# n, bins, patches = plt.hist(x_density, histtype='bar', normed=True, bins=128)
#
# gmm = GaussianMixture(n_components=3, weights_init=[1 / 3, 1 / 3, 1 / 3])
# gmm.fit(x_density)


# def gaussian(x, mu, sig, weigth):
#     return (np.exp(-np.power(x - mu, 2.) / (2 * sig)) / (math.sqrt(2 * math.pi) * math.sqrt(sig))) * weigth


# mu = gmm.means_.flatten()
# sig = gmm.covariances_.flatten()
# combined = []
# for i in range(0, len(mu)):
#     combined.append([mu[i], sig[i], gmm.weights_[i]])
# gauses = []
# for v in combined:
#     g = gaussian(np.arange(w), v[0], v[1], v[2])
#     gauses.append(g)
#
# su = gauses[0] + gauses[1] + gauses[2]
# plt.plot(su)

# minimas = argrelmin(su)
# cv2.line(img, (minimas[0][0], h), (minimas[0][0], 0), (0, 0, 0))
# cv2.line(img, (minimas[0][1], h), (minimas[0][1], 0), (0, 0, 0))
# cv2.imshow("", img)
# cv2.waitKey()
# plt.show()

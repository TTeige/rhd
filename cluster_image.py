import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import root
import math
from scipy.ndimage import affine_transform

imname = "2_27fs10061402171105.jpg"
folder = "fs10061402171105"
full_name = os.path.join("extracted_fields_samples", folder, imname)
img = cv2.imread(full_name, 0)
# img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
w, h = img.shape

np.random.seed(0)
affine = np.array([[1, 0, 0], [-0.1, 1, 0], [0, 0, 1]])
img = affine_transform(img, affine, cval=255)
img_flat = img.flatten()
# img_flat = [print(v) if v == None else v for v in img_flat]
img_flat = [v / 255 for v in img_flat]

x_density = []
for i in range(0, len(img_flat)):
    if img_flat[i] < 0.5:
        x_density.append(np.array([i % w]))

x_density = np.array(x_density)
n, bins, patches = plt.hist(x_density, histtype='bar', density=True, bins=256)

gmm = GaussianMixture(n_components=3, weights_init=[1 / 3, 1 / 3, 1 / 3])
gmm.fit(x_density)


def gaussian(x, mu, sig, weigth):
    return (np.exp(-np.power(x - mu, 2.) / (2 * sig)) / (math.sqrt(2 * math.pi) * math.sqrt(sig))) * weigth


mu = gmm.means_.flatten()
sig = gmm.covariances_.flatten()
combined = []
for i in range(0, len(mu)):
    combined.append([mu[i], sig[i], gmm.weights_[i]])
gauses = []
for v in combined:
    g = gaussian(range(0, w), v[0], v[1], v[2])
    gauses.append(g)

su = gauses[0] + gauses[1] + gauses[2]
plt.plot(su)

from scipy.signal import argrelmin

minimas = argrelmin(su)
cv2.line(img, (minimas[0][0], h), (minimas[0][0], 0), (0, 0, 0))
cv2.line(img, (minimas[0][1], h), (minimas[0][1], 0), (0, 0, 0))
cv2.imshow("", img)
cv2.waitKey()
plt.show()

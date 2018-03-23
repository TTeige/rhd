import cv2
import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from time import time


def read_images(paths):
    images = []
    for p in paths:
        img = cv2.imread(p, 0)
        images.append(normalize_image(img))
    return images


def create_cluster(images, labels):
    h = 0.02
    data = scale(images)

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
    kmeans.fit(reduced_data)
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    img = cv2.imread("/mnt/remote/Yrke/siffer/6_27fs10061408003509/0_6_27fs10061408003509.jpg", 0)
    img = normalize_image(img)
    Z = kmeans.predict(img.reshape([1, [100, 100]]))
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plot_clustering(Z, kmeans, reduced_data, x_max, x_min, xx, y_max, y_min, yy)


def plot_clustering(Z, kmeans, reduced_data, x_max, x_min, xx, y_max, y_min, yy):
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def normalize_image(img):
    img_flat = img.flatten()
    img_flat = [v / 255 for v in img_flat]
    img_flat = np.array(img_flat)
    return img_flat


def bench_k_means(estimator, name, data, labels):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=827)))


def run():
    data = []
    with open("base_images.csv") as f:
        reader = csv.DictReader(f, ["filename", "label"])
        next(reader)
        for row in reader:
            data.append((row["filename"], row["label"]))
    image_paths = [v[0] for v in data]
    labels = [v[1] for v in data]
    images = read_images(image_paths)
    create_cluster(images, labels)


if __name__ == '__main__':
    run()

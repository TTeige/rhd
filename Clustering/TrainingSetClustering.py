import cv2
import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from Database.dbHandler import DbHandler


def restore_image(img, width, height):
    bytearray = np.fromstring(img, np.uint8)
    return bytearray.reshape([width, height])


def read_images(paths):
    images = []
    for p in paths:
        img = cv2.imread(p, 0)
        images.append(normalize_image(img))
    return images


def transform_cluster(images, names, cluster):
    # function assumes indexes are kept after PCA
    data = scale(images)
    reduced_data = PCA(n_components=2).fit_transform(data)
    cluster.transform(reduced_data)
    accepted_distance = 50
    accepted_images = []
    for i, val in enumerate(reduced_data):
        for dist in val:
            if dist < accepted_distance:
                accepted_images.append((names[i], images[i]))

    return accepted_images


def create_cluster(images):
    data = scale(images)
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
    kmeans.fit(reduced_data)

    return kmeans


def normalize_image(img):
    img_flat = img.flatten()
    img_flat = [v / 255 for v in img_flat]
    img_flat = np.array(img_flat)
    return img_flat


def run():
    data = []
    # Read in the base images, should consist of only good quality digits
    with open("base_images.csv") as f:
        reader = csv.DictReader(f, ["filename", "label"])
        next(reader)
        for row in reader:
            data.append((row["filename"], row["label"]))
    image_paths = [v[0] for v in data]
    # labels = [v[1] for v in data]
    images = read_images(image_paths)
    # Create the cluster based on these images.
    cluster = create_cluster(images)

    filename_list = []

    with open("cp_joined.csv") as f:
        reader = csv.DictReader(f, ["filename", "label"])
        next(reader)
        for row in reader:
            filename_list.append(row["filename"])

    # Feed new images to the cluster
    with DbHandler("ft1950_ml.db") as db:
        with open("new_training.txt", "w") as f:
            images = []
            for i, entry in enumerate(db.select_all_images()):
                if entry[0] not in filename_list:
                    continue
                images.append((entry[0], entry[1]))
                if i % 1000 == 0 and i != 0:
                    close_enough_imgs = transform_cluster(images[1], images[0], cluster)
                    for name in close_enough_imgs[0]:
                        f.write(name)
                    images = []


if __name__ == '__main__':
    run()

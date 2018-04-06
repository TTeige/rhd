import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import math
from scipy.ndimage import affine_transform
from scipy.signal import argrelmin, argrelmax
import concurrent.futures as cf
import time
import argparse
from Database.dbHandler import DbHandler


class GaussianNormalDistributionCluster:
    """
    GaussianNormalDistributionCluster provides methods for extracting the density distribution of an image,
    it's summed gaussian normal distributions and it's minimas for digit seperation.
    In order to render the plots, matplotlib.pyplot.show() must be called after the rendering methods are called.
    The load_image(path) method must be called before using any other method.
    """

    def __init__(self, num_components=3):
        """
        :param num_components: number of gaussian normal distributions
        :param img: image to process
        """
        self.image = None
        self.components = num_components
        self.shape = (100, 100)
        self.gaussian_values = None

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

    def load_image(self, img, width, height):
        """
        Loads an image in grayscale using opencv
        :param img: image in byte values
        :return: ndarray of pixel values, grayscale
        :type:ndarray
        """

        bytearray = np.fromstring(img, np.uint8)
        self.image = bytearray.reshape([width, height])
        affine = np.array([[1, 0, 0], [-0.3, 1, 0], [0, 0, 1]])
        img = affine_transform(self.image, affine, cval=255)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        self.image = img
        if self.image is None:
            raise ValueError("Unable to load image, check path")
        return self.image

    def get_x_density(self):
        """
        Creates a 1d array containing the location of pixel values on the x-axis above a threshold,
        load_image must be called first
        :return: list of pixel locations
        """
        if self.image is None:
            raise ValueError
        rows, cols = self.image.shape

        np.random.seed(0)
        img_flat = self.image.flatten()
        img_flat = [v / 255 for v in img_flat]
        img_flat = np.array(img_flat)
        x_density = []
        for i in range(0, len(img_flat)):
            if img_flat[i] < 0.2:
                x_density.append(np.array([i % cols]))

        return np.array(x_density)

    def get_minimas(self, summed_gaussian=None):
        """
        Returns local minimas of the gaussian function
        :param summed_gaussian: sum of gaussian normal distributions. If None, the method will retrieve a summed
        gaussian for the given number of components
        :return: local minimas. None if the image contains no valid pixels, see method get_x_density().
        """
        if summed_gaussian is None:
            summed_gaussian = self.get_summed_gaussian()
            if summed_gaussian is None:
                return None
        minims = argrelmin(summed_gaussian)
        return minims

    def get_maxims(self, summed_gaussian=None):
        """
        Finds the maximum points for the summed gaussian function. Can handle single gaussian functions as well.
        :param summed_gaussian: Function of which to find the local maximum
        :return: array of local maximum values
        """
        if summed_gaussian is None:
            summed_gaussian = self.get_summed_gaussian()
            if summed_gaussian is None:
                return None
        maxims = argrelmax(summed_gaussian)
        return maxims

    @staticmethod
    def render_hist(x_density, num_bins=28):
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
        :return: summed gaussian distribution. If None, no valid (normalized pixels < 0.1) pixels are in the image
        """

        if x_density is None:
            x_density = self.get_x_density()

        if len(x_density) == 0:
            return None

        weights = np.full(self.components, init_weight)
        gmm = GaussianMixture(n_components=self.components, weights_init=weights)
        gmm.fit(x_density)

        mu = gmm.means_.flatten()
        sig = gmm.covariances_.flatten()
        gausses = []
        for i in range(0, len(mu)):
            g = self.gaussian(np.arange(self.image.shape[1]), mu[i], sig[i], gmm.weights_[i])
            gausses.append(g)
        gausses = np.array(gausses)
        self.gaussian_values = gausses
        sum_g = gausses.sum(axis=0)
        return sum_g

    def resize_images(self, images):
        completed = []
        for image in images:
            if image.shape[0] == 0:
                print("The image shape on the x axis is {}".format(image.shape[0]))
            if image.shape[1] == 0:
                print("The image shape on the y axis is {}".format(image.shape[1]))
            if image.shape[0] > self.shape[0]:
                # Resize the image if an axis is too large to fit in the new image
                if image.shape[1] > self.shape[1]:
                    # Both axis in the image is greater than the wanted shape, resize both axis
                    image = cv2.resize(image, self.shape, interpolation=cv2.INTER_CUBIC)
                else:
                    # Only the X axis is greater, resize only this
                    image = cv2.resize(image, (image.shape[1], self.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                if image.shape[1] > self.shape[1]:
                    # Only the Y axis is greater, resize only this
                    image = cv2.resize(image, (self.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

            reshaped = np.full(self.shape, 0, dtype='uint8')
            p = np.array(image)
            x_offset = int(abs(image.shape[0] - self.shape[0]) / 2)
            y_offset = int(abs(image.shape[1] - self.shape[1]) / 2)
            reshaped[x_offset:p.shape[0] + x_offset, y_offset:p.shape[1] + y_offset] = p
            completed.append(reshaped)

        return completed

    def split_image(self, image, split_points, mid_points):
        """
        Splits the image based on the location of the minimum points given by the summed gaussian function
        :param image: Input image in grayscale
        :param split_points: Local minimum points of the summed gaussian function
        :param mid_points: Maximum points of the summed gaussian function
        :return: an array of the split images
        """
        image = cv2.bitwise_not(image)
        new1 = np.array([row[:split_points[0]] for row in image])
        new2 = np.array([row[split_points[0]:split_points[1]] for row in image])
        new3 = np.array([row[split_points[1]:] for row in image])

        def test_for_value(col):
            for col_val in col:
                if col_val > 200:
                    # We found a value in this column, so go to next
                    return True
            return False

        try:
            new1 = self.reshape_left_image(new1, test_for_value, mid_points[0])
        except ValueError as e:
            try:
                intersections = self.find_intersections()
                new1 = np.array([row[:intersections[0]] for row in image])
                new1 = self.reshape_left_image(new1, test_for_value, mid_points[0])
            except Exception as e:
                print("Left image has wrong shape {}, exception: {}".format(new1.shape, e))
                return None

        try:
            new2 = self.reshape_middle_image(new2)
        except ValueError as e:
            try:
                intersections = self.find_intersections()
                new2 = np.array([row[intersections[0]:intersections[1]] for row in image])
                new2 = self.reshape_middle_image(new2)
            except Exception as e:
                print("Middle image has wrong shape {}, exception: {}".format(new2.shape, e))
                return None

        try:
            new3 = self.reshape_right_image(new3, test_for_value, mid_points[2] - split_points[1])
        except ValueError as e:
            try:
                intersections = self.find_intersections()
                new3 = np.array([row[intersections[1]:] for row in image])
                new3 = self.reshape_right_image(new3, test_for_value, mid_points[2] - intersections[1])
            except Exception as e:
                print("Right image has wrong shape {}, exception: {}".format(new3.shape, e))
                return None
        all_i = [new1, new2, new3]

        return self.resize_images(all_i)

    @staticmethod
    def reshape_right_image(new3, test_for_value, digit_center_point):
        # Right image
        # Calculate offset from the total image length
        from_mid = np.swapaxes(new3[:, digit_center_point:], 1, 0)
        for i in range(0, from_mid.shape[0] - 2, 2):
            # Iterate from the top of the new image
            # Check if the row contains values
            if not test_for_value(from_mid[i]):
                # Check the next row for values
                if not test_for_value(from_mid[i + 1]) and not test_for_value(from_mid[i + 2]):
                    # We found a row without values, and the next does not either
                    # Copy over the values based on the new first column containing values
                    new3 = new3[:, :i + digit_center_point]
                    break
        if new3.shape[0] == 0 or new3.shape[1] == 0:
            raise ValueError
        return new3

    @staticmethod
    def reshape_middle_image(new2):
        # left = self.reshape_left_image(new2, test_for_value, digit_center_point)
        # right = self.reshape_right_image(new2, test_for_value, digit_center_point)
        # if left.shape[0] < right.shape[0]:
        #     new2 = left
        # else:
        #     new2 = right
        if new2.shape[0] == 0 or new2.shape[1] == 0:
            raise ValueError
        return new2

    @staticmethod
    def reshape_left_image(new1, test_for_value, digit_center_point):
        # Left image
        # Extract array from mid point of the digit and switch to column major order
        from_mid = np.swapaxes(new1[:, digit_center_point:0:-1], 1, 0)
        for i in range(0, from_mid.shape[0] - 2, 2):
            # Iterate from the bottom of the new image
            # Check if the row contains values
            if not test_for_value(from_mid[i]):
                # Check the next row for values
                if not test_for_value(from_mid[i + 1]) and not test_for_value(from_mid[i + 2]):
                    # We found a row without values, and the next does not either
                    # Copy over the values based on the new first column containing values
                    new1 = new1[:, digit_center_point - i:]
                    break
        if new1.shape[0] == 0 or new1.shape[1] == 0:
            raise ValueError
        return new1

    def find_intersections(self):
        """
        Finds the intersection between the gaussian functions. These are loaded from the class and assumes that the
        gaussian functions have already been created. Fails with an exception by default if the functions are not
        created
        :return:
        """
        gaus_and_mid = []
        for val in self.gaussian_values:
            gaus_and_mid.append((self.get_maxims(val)[0][0], val))
        gaus_and_mid = sorted(gaus_and_mid, key=lambda q: q[0])
        intersections = []
        try:
            for i in range(0, len(gaus_and_mid) - 1):
                for k, val in enumerate(gaus_and_mid[i][1]):
                    if k == len(gaus_and_mid[i][1]) - 3:
                        break
                    a = val
                    b = gaus_and_mid[i + 1][1][k]
                    c = gaus_and_mid[i][1][k + 3]
                    d = gaus_and_mid[i + 1][1][k + 3]
                    if a > c:
                        tmp = c
                        c = a
                        a = tmp
                    if b > d:
                        tmp = d
                        d = b
                        b = tmp
                    if (a <= d and c >= b) and k > gaus_and_mid[i][0]:
                        intersections.append(k)
                        break
        except Exception as e:
            print(e)
        return intersections


def run_test(db_loc, image_name="4_27fs10061402170627.jpg"):
    """
    Test run against single images
    :param path: path to the image
    :return:
    """
    # np.random.seed(0)
    db = DbHandler(db_loc)
    db_image_entry = db.select_image(image_name)
    gnc = GaussianNormalDistributionCluster()
    img = gnc.load_image(db_image_entry[1], db_image_entry[2], db_image_entry[3])
    x_density = gnc.get_x_density()
    gnc.render_hist(x_density)
    sum_g = gnc.get_summed_gaussian(x_density)
    gnc.render_dist(sum_g)
    mins = gnc.get_minimas(sum_g)
    maxes = gnc.get_maxims(sum_g)
    plt.scatter(np.append(mins[0], maxes[0]), np.append(sum_g[mins[0]], sum_g[maxes[0]]), c='r', zorder=10)
    plt.show()
    new_images, _, _ = execute("", db_image_entry[1], db_image_entry[2], db_image_entry[3])

    cv2.line(gnc.image, (mins[0][0], img.shape[1]), (mins[0][0], 0), 0)
    cv2.line(gnc.image, (mins[0][1], img.shape[1]), (mins[0][1], 0), 0)

    cv2.imshow("0", new_images[0])
    cv2.imshow("1", new_images[1])
    cv2.imshow("2", new_images[2])
    cv2.imshow("image", gnc.image)
    cv2.waitKey()


def execute(name, img, height, width):
    """
    Function to handle the launching of a parallel task
    :param name: Name of the image
    :param img: image
    :return: list of images separated, name of the file, error message if not completed
    """
    gnc = GaussianNormalDistributionCluster()
    try:
        image = gnc.load_image(img, width, height)
        x_density = gnc.get_x_density()
        sum_g = gnc.get_summed_gaussian(x_density)
        mins = gnc.get_minimas(sum_g)
        if mins is None:
            return None, name, "No minimums found"
        maxes = gnc.get_maxims(sum_g)
        if maxes is None:
            return None, name, "No maximums found"
    except ValueError as e:
        # Unsure of what exactly happens here, but the x_density vector is only a single dimension
        # which causes the GMM to fail. This can happen if there is only a single row containing pixels, or none
        # These images are however not relevant and can be skipped.

        print("{} Skipping image at path: {} due to lacking values in x_density".format(e, name))
        return None, name, " lacking values in x_density. Exception {}".format(e)

    try:
        new_images = gnc.split_image(image, mins[0], maxes[0])
        if new_images is None:
            return None, name, "No images returned"
        return new_images, name, ""
    except IndexError as e:
        # Only one minima is found, this is the wrong result for the profession field. Should be two minimas
        # So these images are just skipped.
        print("{} Skipping image at path: {} due to single minima or maxima".format(e, name))
        return None, name, "single minima or maxima. Exception {}".format(e)


def handle_done(done, db):
    """
    Function to handle the output of a parallel task
    :param done: Handle to the result
    :type: Future
    :param db: database handler
    :type: DbHandler
    :return:
    """
    new_images, name, err = done.result()
    if new_images is None or err != "":
        try:
            db.store_dropped(name, err)
        except Exception as e:
            print(e)
    else:
        for i, im in enumerate(new_images):
            name = str(i) + "_" + name
            try:
                db.store_digit(name, im)
            except Exception as e:
                print(e)


def run_parallel(db_loc):
    """
    Launches the parallel executor and submits all the jobs. This function parses the entire folder structure and keeps
    it in memory
    :param db_loc: database location, full path
    :return:
    """
    np.random.seed(0)
    start_time = time.time()
    futures = []
    num = 0
    # The following is not big data safe, could run out of memory, best would be to create a stream, but python....
    image_strings = []
    # All usage of image strings might be volatile
    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        with DbHandler(db_loc) as db:
            num_read = db.count_rows_in_fields()
            for db_img in db.select_all_images():
                futures.append(executor.submit(execute, db_img[0], db_img[1], db_img[2], db_img[3]))

            for done in cf.as_completed(futures):
                handle_done(done, db)
                futures.remove(done)
                num += 1
                if num % 100 == 0:
                    print("Number of images segmented is: {} out of a total of {}".format(num, num_read))
                    db.connection.commit()
            print("--- " + str(time.time() - start_time) + " ---")


def handle_main():
    arg = argparse.ArgumentParser("Extract individual digits from image")
    arg.add_argument("-t", "--test", action="store_true", default=False, help="Run the program in test_mode")
    arg.add_argument("--db", type=str, help="full path to database location",
                     default="/mnt/remote/Yrke/ft1950_ml.db")
    args = arg.parse_args()
    if args.test:
        run_test(args.db)
    else:
        run_parallel(args.db)


if __name__ == '__main__':
    handle_main()

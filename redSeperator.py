import cv2
import numpy as np
import concurrent.futures as cf
import time
import os


def handleImage(fn):
    img = cv2.imread(fn)
    img = resize_img(img)
    img = convert_img(img)
    cv2.imwrite("numbers/" + fn, img)


def resize_img(img):
    height, width = img.shape[:2]
    return cv2.resize(img, (int(0.3 * width), int(0.3 * height)), interpolation=cv2.INTER_CUBIC)


def convert_img(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 45, 50])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([160, 50, 50])
    upper_red = np.array([190, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    mask = mask0 + mask1

    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0

    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0
    gray_channel = cv2.split(output_hsv)[2]
    return gray_channel


if __name__ == '__main__':
    from glob import glob

    start_time = time.time()
    completed_images = 0
    cwd = os.getcwd()
    if not os.path.exists(cwd + "/numbers"):
        os.makedirs(cwd + "/numbers")
    with cf.ProcessPoolExecutor() as executor:
        futures = []
        for i, fn in enumerate(glob('006KID67886_K0-3/*')):
            print(fn)
            if i == 10:
                break
            futures.append(executor.submit(handleImage, fn))
        for done in cf.as_completed(futures):
            completed_images += 1
            if completed_images % 10 == 0:
                print("Progress: " + str(completed_images / len(futures) * 100) + "%")
    print("--- %s seconds ---" % (time.time() - start_time))

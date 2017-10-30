import cv2
import numpy as np
import concurrent.futures as cf
import time
import os
import csv

import sys


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def find_box_area(x1, y1, x2, y2):
    return (x2 - x1 + 1) * (y2 - y1 + 1)


def truncate_float(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])


# def find_if_close(cnt1, cnt2):
#     row1, row2 = cnt1.shape[0], cnt2.shape[0]
#     for i in range(row1):
#         for j in range(row2):
#             dist = np.linalg.norm(cnt1[i] - cnt2[j])
#             if abs(dist) < 2:
#                 return True
#             elif i == row1 - 1 and j == row2 - 1:
#                 return False
#
#
# def non_max_suppression_fast(boxes, overlapThresh):
#     # if there are no boxes, return an empty list
#     if len(boxes) == 0:
#         return []
#
#     # if the bounding boxes integers, convert them to floats --
#     # this is important since we'll be doing a bunch of divisions
#     if boxes.dtype.kind == "i":
#         boxes = boxes.astype("float")
#
#     # initialize the list of picked indexes
#     pick = []
#
#     # grab the coordinates of the bounding boxes
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#
#     # compute the area of the bounding boxes and sort the bounding
#     # boxes by the bottom-right y-coordinate of the bounding box
#     area = (x2 - x1 + 1) * (y2 - y1 + 1)
#     idxs = np.argsort(y2)
#
#     # keep looping while some indexes still remain in the indexes
#     # list
#     while len(idxs) > 0:
#         # grab the last index in the indexes list and add the
#         # index value to the list of picked indexes
#         last = len(idxs) - 1
#         i = idxs[last]
#         pick.append(i)
#
#         # find the largest (x, y) coordinates for the start of
#         # the bounding box and the smallest (x, y) coordinates
#         # for the end of the bounding box
#         xx1 = np.maximum(x1[i], x1[idxs[:last]])
#         yy1 = np.maximum(y1[i], y1[idxs[:last]])
#         xx2 = np.minimum(x2[i], x2[idxs[:last]])
#         yy2 = np.minimum(y2[i], y2[idxs[:last]])
#
#         # compute the width and height of the bounding box
#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)
#
#         # compute the ratio of overlap
#         overlap = (w * h) / area[idxs[:last]]
#
#         # delete all indexes from the index list that have
#         idxs = np.delete(idxs, np.concatenate(([last],
#                                                np.where(overlap > overlapThresh)[0])))
#
#     # return only the bounding boxes that were picked using the
#     # integer data type
#     return boxes[pick].astype("int")
#
#
# def join_contours(contours):
#     length = len(contours)
#     status = np.zeros((length, 1))
#     for i, cnt1 in enumerate(contours):
#         x = 1
#         if i != length - 2:
#             for j, cnt2 in enumerate(contours[i + 1:]):
#                 x = x + 1
#                 dist = find_if_close(cnt1, cnt2)
#                 if dist:
#                     val = min(status[i], status[x])
#                     status[x] = status[i] = val
#                 else:
#                     if status[x] == status[i]:
#                         status[x] = i + 1
#
#     unified = []
#     maximum = int(status.max()) + 1
#     for i in range(maximum):
#         pos = np.where(status == i)[0]
#         if pos.size != 0:
#             cont = np.vstack(contours[i] for i in pos)
#             hull = cv2.convexHull(cont)
#             unified.append(hull)
#     return unified


def handle_image(target_dir, file_location, fn):
    try:
        img_original = cv2.imread(file_location)
        # img = resize_img(img_original)
        img = convert_img(img_original)
        kernel = np.ones((5, 5), np.uint8)
        retval, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        img = cv2.dilate(img, kernel, iterations=1)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)
        img2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        single_image_dir = os.path.join(target_dir, fn.split('.')[0])
        if not os.path.exists(single_image_dir):
            os.makedirs(single_image_dir)

        with open(os.path.join(single_image_dir, fn.split('.')[0] + "_bb_locations.csv"), "w") as csv_file:
            writer = csv.DictWriter(csv_file, ['fn', 'x', 'y', 'w', 'h'])
            writer.writeheader()
            # bounding_rects = []
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                rect = Rect(x, y, w, h)
                # bounding_rects.append(rect)

                # completed_boxes = []

                # Make a naive assumption  that rectangles can only be merged once
                # Have to work around the edge cases of poorly written 5 numbers
                # for rect in bounding_rects:
                #     for rect2 in bounding_rects:
                #         if rect == rect2:
                #             continue
                #         if check_merge(rect, rect2):
                #             merged = merge_rects(rect, rect2)
                #             completed_boxes.append(merged)
                #             bounding_rects.remove(rect2)
                #             bounding_rects.remove(rect)
                #             break
                #
                # for rect in bounding_rects:
                #     completed_boxes.append(rect)

                # for i, rect in enumerate(completed_boxes):
                # if not validate_box(rect.x, rect.y, rect.w, rect.h):
                #     continue
                candidate = extract_number(rect.x, rect.y, rect.w, rect.h, img)
                reshaped = reshape_img(candidate)
                fn_final = os.path.join(single_image_dir, (str(i) + "_" + fn))
                writer.writerow({'fn': fn_final, 'x': int(rect.x / 0.3), 'y': int(rect.y / 0.3),
                                 'w': int(rect.w / 0.3), 'h': int(rect.h / 0.3)})
                # writer.writerow({'fn': fn_final, 'x': int(rect.x), 'y': int(rect.y),
                #                  'w': int(rect.w), 'h': int(rect.h)})
                cv2.imwrite(fn_final, reshaped)
        return fn

    except Exception as e:
        print(e)


def merge_rects(rect1, rect2):
    if rect1.x <= rect2.x:
        new_x = rect1.x
    else:
        new_x = rect2.x

    if rect1.y <= rect2.y:
        new_y = rect1.y
    else:
        new_y = rect2.y

    if rect1.x + rect1.w <= rect2.x + rect2.w:
        new_x2 = rect2.x + rect2.w
    else:
        new_x2 = rect1.x + rect1.w

    if rect1.y + rect1.h <= rect2.y + rect2.h:
        new_y2 = rect2.y + rect2.h
    else:
        new_y2 = rect1.y + rect1.h

    new_w = new_x2 - new_x
    new_h = new_y2 - new_y

    return Rect(new_x, new_y, new_w, new_h)


def validate_box(x, y, w, h):
    if w * h < 50 or w * h > 3000:
        return False
    if w / h > 5 or h / w > 5:
        return False
    return True


def calculate_distance(rect1, rect2):
    a = np.array([rect1.x, rect1.y])
    b = np.array([rect2.x, rect2.y])
    return np.linalg.norm(a - b)


def check_merge(rect1, rect2):
    # Check if the merge target is between leftmost and rightmost edge of the bounding box
    if rect1.x - 10 <= rect2.x <= rect1.x + rect1.w + 10:
        # Only merge if box is above and within threshold
        if np.abs(rect1.y - rect2.y) < 10 and rect2.h < 10:
            return True
    return False


def extract_number(x, y, w, h, img):
    possible_number_img = img[y:y + h, x:x + w]
    possible_number_img = cv2.resize(possible_number_img, (24, 24), interpolation=cv2.INTER_AREA)
    possible_number_img = cv2.bitwise_not(possible_number_img)
    # retval, possible_number_img = cv2.threshold(possible_number_img, 150, 255, cv2.THRESH_BINARY)
    return possible_number_img


def reshape_img(possible_number_img):
    reshaped = np.full((28, 28), 255, dtype='uint8')
    p = np.array(possible_number_img)
    x_off = y_off = 2
    reshaped[x_off:p.shape[0] + x_off, y_off:p.shape[1] + y_off] = p
    return reshaped


def resize_img(img):
    height, width = img.shape[:2]
    return cv2.resize(img, (int(width * 0.3), int(height * 0.3)), interpolation=cv2.INTER_CUBIC)


def convert_img(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask
    lower_red = np.array([0, 45, 50])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask
    lower_red = np.array([160, 50, 50])
    upper_red = np.array([190, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    mask = mask0 + mask1

    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0
    gray_channel = cv2.split(output_hsv)[2]
    return gray_channel


def main(image_path, output_path="./out"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    start_time = time.time()
    with cf.ProcessPoolExecutor() as executor:
        futures = []
        completed_images = 0
        total_files = 0
        with open("progress", "a+") as progress_file:
            progress_file.seek(0)
            completed_files = {}
            for line in progress_file:
                line = line.rstrip()
                completed_files[line] = True

            for dirname, dirnames, filenames in os.walk(image_path):
                total_files += len(filenames)
                target_path = os.path.join(output_path, dirname.split("/")[-1])
                print(target_path)
                for i, fn in enumerate(filenames):
                    if fn.split(".")[1] != "jpg":
                        continue
                    if completed_files.get(fn):
                        print("Skipping " + fn)
                        completed_images += 1
                        continue
                    file_location = os.path.join(dirname, fn)
                    futures.append(
                        executor.submit(handle_image, target_path, file_location, fn))
                for done in cf.as_completed(futures):
                    fn = done.result()
                    progress_file.write(fn + '\n')
                    futures.remove(done)
                    completed_images += 1
                    if completed_images % 10 == 0:
                        print("--- Current duration: %s seconds ---" % (truncate_float(time.time() - start_time, 3)))
                        print("Progress: " + truncate_float(str(completed_images / total_files * 100), 3) + "%"
                              + " Images completed: " + str(completed_images))
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 " + __file__ + " <path/to/images>")
        print("Optional: <output/image/path>")
        exit(1)
    if sys.argv[1].lower() == "help":
        print("Usage: python3 " + __file__ + " <path/to/images>")
        print("Optional: <output/image/path>")
        exit(1)
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1])

import cv2

if __name__ == '__main__':
    from glob import glob

    for i, fn in enumerate(glob('006KID67886_K0-3/*')):
        print(fn, i, sep='\t')
        img = cv2.imread(fn)
        height, width = img.shape[:2]
        resized = cv2.resize(img, (int(0.3*width), int(0.3*height)), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("resized_" + fn, resized)

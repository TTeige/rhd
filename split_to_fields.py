import cv2
import sys


def extract_field(img, row_1, row_2, i):
    # x position different index on same row
    x1 = row_1[i][0]
    x2 = row_1[i + 2][0]
    # y position same index on different row
    y1 = row_1[i][1]
    y2 = row_2[i][1]
    field_img = img[y1:y2, x1:x2]
    return field_img


def split_row(img_path, row_1, row_2):
    try:
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        i = 0
        while i + 2 < len(row_1):
            field_img = extract_field(img, row_1, row_2, i)

            cv2.imshow("field", field_img)
            cv2.waitKey()
            i += 2
    except Exception as e:
        print(e)


def get_filename(line):
    # Filename is the first segment after the first comma
    filename = line.split('<')[0].split(',')[1]
    if mod != "":
        filename = filename.split('=')[1]
        filename = mod + filename + ".jpg"

    return filename


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Specify coordinates file")
        print("Usage: python3 path/to/file")
        print("Usage optional: python3 path/to/file image/path/modification")
        print("If the modification path is specified, the share= will be replaced with the modified path")
        exit(1)
    print(sys.argv[1])

    mod = ""

    if len(sys.argv) == 3:
        mod = sys.argv[2]

    num_done = 0

    with open(sys.argv[1], 'r') as co_file:
        # Skip the header line
        next(co_file)

        # Cache the first line since we need it when we read the second line
        first_line = co_file.readline()

        for second_line in co_file:
            filename = get_filename(first_line)

            # Reset the first line to the next image and read a new second line
            if filename != get_filename(second_line):
                first_line = second_line
                second_line = co_file.readline()
                print(filename)

            row_1_str = first_line.split('<')
            row_2_str = second_line.split('<')

            row_1 = []
            row_2 = []

            # Start the coordinate extraction after the row index
            for token in row_1_str[3:]:
                coordinate = token.split(',')
                if len(coordinate) == 2:
                    row_1.append((int(coordinate[0]), int(coordinate[1])))
                else:
                    row_1.append(token)

            for token in row_2_str[3:]:
                coordinate = token.split(',')
                if len(coordinate) == 2:
                    row_2.append((int(coordinate[0]), int(coordinate[1])))
                else:
                    row_2.append(int(token))

            first_line = second_line
            # print("First row index = " + row_1_str[1])
            # print("Second row index = " + str(row_2_str[1]))

            split_row(filename, row_1, row_2)

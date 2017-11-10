import cv2
import argparse
import numpy as np

col_names = {"husholdnings_nr": 0,
             "person_nr": 1,
             "navn": 2,
             "stilling_i_hus": 3,
             "kjonn": 4,
             "fast_bosatt_og_tilstedet": 5,
             "midlertidig_fraværende": 6,
             "midlertidig_tilstedet": 7,
             "fødselsdato": 8,
             "fødested": 9,
             "ekteskapelig_stilling": 10,
             "ekteskaps_aar": 11,
             "barnetall": 12,
             "arbeid": 13,
             "egen_virksomhet": 14,
             "bedrift_arbeidsgiver": 15,
             "arbeidssted": 16,
             "biyrke": 17,
             "hjelper_hovedperson": 18,
             "utdanning_artium": 19,
             "høyere_utdanning": 20,
             "trossamfunn": 21,
             "borgerrett": 22,
             "innflytning": 23,
             "sist_kommune": 24,
             "bosatt_i_1946": 25}


def extract_field(img, row_1, row_2, i):
    # x position different index on same row
    x1 = row_1[i][0]
    x2 = row_1[i + 2][0]
    # y position same index on different row
    y1 = row_1[i][1]
    y2 = row_2[i][1]
    field_img = img[y1:y2, x1:x2]
    return field_img


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
    retval, gray_channel = cv2.threshold(gray_channel, 100, 255, cv2.THRESH_BINARY)
    gray_channel = cv2.GaussianBlur(gray_channel, (3, 3), 0)
    gray_channel = cv2.bitwise_not(gray_channel)
    return gray_channel


def check_extraction(img, row_1, row_2, i, target_fields):
    field_img = []
    if len(target_fields) > 0:
        if row_1[i - 1] in target_fields:
            field_img = extract_field(img, row_1, row_2, i)
        elif isinstance(target_fields[0], str):
            for name in target_fields:
                if col_names[name] == row_1[i - 1]:
                    field_img = extract_field(img, row_1, row_2, i)

    else:
        field_img = extract_field(img, row_1, row_2, i)

    return field_img


def split_row(img_path, row_1, row_2, target_fields, process_images):
    try:
        fields = []
        img = cv2.imread(img_path)
        if process_images:
            img = convert_img(img)
        i = 1
        while i + 2 < len(row_1):
            field_img = check_extraction(img, row_1, row_2, i, target_fields)
            if len(field_img) != 0:
                fields.append(field_img)
                cv2.imshow("", field_img)
                cv2.waitKey()
            i += 2
        return fields
    except Exception as e:
        print(e)


def get_filename(line, mod):
    # Filename is the first segment after the first comma
    filename = line.split('<')[0].split(',')[1]
    if mod != "":
        filename = filename.split('=')[1]
        filename = mod + filename + ".jpg"

    return filename


def extract_row(line):
    _row = line.split('<')

    row = []
    # Start the coordinate extraction after the column index
    for token in _row[2:]:
        coordinate = token.split(',')
        if len(coordinate) == 2:
            row.append((int(coordinate[0]), int(coordinate[1])))
        else:
            row.append(int(token))

    # Returns the row and the row index
    return row, int(_row[1])


def main(args):
    mod = args.img_path_mod

    print("Image: " + args.coordinate_file)
    print("Replacement prefix: " + mod)

    with open(args.coordinate_file, 'r') as co_file:
        # Skip the header line
        next(co_file)

        if args.type == "digits":
            next(co_file)

        first_line = co_file.readline()

        for second_line in co_file:
            filename = get_filename(first_line, mod)

            # Reset the first line to the next image and read a new second line
            if filename != get_filename(second_line, mod):
                first_line = second_line
                second_line = co_file.readline()

            row_1, row_1_index = extract_row(first_line)
            row_2, row_2_index = extract_row(second_line)
            first_line = second_line

            if args.type == "digits":
                if row_2_index % 2 != 0:
                    continue
            elif args.type == "writing":
                if row_2_index % 2 == 0:
                    continue

            target_fields = []
            if len(args.cols_number) != 0:
                target_fields = args.cols_number
            elif len(args.cols_name) != 0:
                target_fields = args.cols_name

            row = split_row(filename, row_1, row_2, target_fields, args.process_images)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description="Extract fields from the given image using precalculated coordinates")
    arg_parser.add_argument("coordinate_file", type=str, help="path to the coordinate file")
    arg_parser.add_argument("--img_path_mod", "-m", metavar="M", type=str, default="",
                            help="replaces the share= in the coordinate file and replaces it with the given argument")
    arg_parser.add_argument("--type", "-t", metavar="T", type=str,
                            help="specifies the type of fields to be extracted", choices=["all", "digits", "writing"],
                            default="all")
    arg_parser.add_argument("--process_images", "-P", action="store_true", help="convert and filter images")
    arg_parser.add_argument("--cols_number", "-c", nargs='+',
                            help="specifies the column index for which to search for. "
                                 "Defined in a space separated list 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26",
                            default=[])
    arg_parser.add_argument("--cols_name", "-C", nargs='+',
                            help="specifies the column by name, defined as a list of names separated by spaces"
                                 "husholdnings_nr  "
                                 "person_nr  "
                                 "navn "
                                 "stilling_i_hus "
                                 "kjønn "
                                 "fast_bosatt_og_tilstedet "
                                 "midlertidig_fraværende "
                                 "midlertidig_tilstedet "
                                 "fødselsdato "
                                 "fødested "
                                 "ekteskapelig_stilling "
                                 "ekteskaps_aar "
                                 "barnetall "
                                 "arbeid "
                                 "egen_virksomhet "
                                 "bedrift_arbeidsgiver "
                                 "arbeidssted "
                                 "biyrke "
                                 "hjelper_hovedperson "
                                 "utdanning_artium "
                                 "høyere_utdanning "
                                 "trossamfunn "
                                 "borgerrett "
                                 "innflytning "
                                 "sist_kommune "
                                 "bosatt_i_1946",
                            default=[])

    args = arg_parser.parse_args()
    _tmp = []
    for num in args.cols_number:
        _tmp.append(int(num))
    args.cols_number = _tmp
    main(args)

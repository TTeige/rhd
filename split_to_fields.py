import cv2
import argparse
import numpy as np
import os
import concurrent.futures as cf
import time

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
    for i in range(0, gray_channel.shape[1]):
        done = False
        for col in gray_channel[:, i]:
            if col != 0:
                gray_channel = np.delete(gray_channel, np.s_[:i], axis=1)
                done = True
                break
        if done:
            break
    for i in range(gray_channel.shape[1] - 1, 0, -1):
        done = False
        for col in gray_channel[:, i]:
            if col != 0:
                gray_channel = np.delete(gray_channel, np.s_[i:], axis=1)
                done = True
                break
        if done:
            break

    gray_channel = cv2.bitwise_not(gray_channel)

    gray_channel = cv2.resize(gray_channel, (60, 60), interpolation=cv2.INTER_AREA)

    reshaped = np.full((64, 64), 255, dtype='uint8')
    p = np.array(gray_channel)
    x_off = y_off = 2
    reshaped[x_off:p.shape[0] + x_off, y_off:p.shape[1] + y_off] = p

    return reshaped


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
        if len(img) == 0:
            print("Image not found: " + img_path + " , check path prefix or remote connections")
            return []

        for i in range(1, len(row_1) - 2, 2):
            field_img = check_extraction(img, row_1, row_2, i, target_fields)
            if len(field_img) != 0:
                if process_images:
                    field_img = convert_img(field_img)
                fields.append((field_img, i))
        return fields
    except Exception as e:
        print("Skipping image " + img_path + " Error: ")
        print(e)


def get_filename(line, mod):
    filename = ""
    if line == "":
        return filename
    try:
        # Filename is the first segment after the first comma
        filename = line.split('<')[0].split(',')[1]
        if mod != "":
            filename = filename.split('=')[1]
            filename = mod + filename + ".jpg"
    except Exception as e:
        print(e)

    return filename


def split_row_str(line):
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


def read_full_image_lines(file, mod, first):
    first_line = first

    rows = []
    while True:

        second_line = file.readline()
        filename = get_filename(first_line, mod)
        filename2 = get_filename(second_line, mod)
        rows.append(first_line)
        if filename != filename2:
            return rows, filename, second_line

        # rows.append(second_line)
        first_line = second_line


def process_rows(filename, rows, args):
    extracted_rows = extract_rows(rows, args)

    target_fields = []
    if len(args.cols_number) != 0:
        target_fields = args.cols_number
    elif len(args.cols_name) != 0:
        target_fields = args.cols_name

    image_fields = []
    for i in range(0, len(extracted_rows) - 1, 2):
        row_1 = extracted_rows[i][0]
        row_2 = extracted_rows[i + 1][0]
        fields = split_row(filename, row_1, row_2, target_fields, args.process_images)
        i += 1
        image_fields.append((fields, extracted_rows[i][1]))
    return image_fields, filename


def extract_rows(rows, args):
    index = 0
    step = 1
    if args.type == "digits":
        step = 2
        index = 1
    elif args.type == "writing":
        step = 2
    extracted_row = []
    for k in range(index, len(rows) - 1, step):
        row_1, row_1_index = split_row_str(rows[k])
        row_2, row_2_index = split_row_str(rows[k + 1])
        extracted_row.append((row_1, row_1_index))
        extracted_row.append((row_2, row_2_index))

    return extracted_row


def create_img_list(img_range):
    propper_img_list = []
    count = 0
    # img_list_start = ["fs10061402170436"]
    # img_list_end = ["fs10061402177225"]
    img_list_start = img_range[0]
    img_list_end = img_range[1]
    start_num = int(img_list_start[0].split("fs")[-1])
    end_num = int(img_list_end[0].split("fs")[-1])
    for index in img_range(start_num, end_num):
        propper_img_list.append("fs" + str(index))
        count += 1

    return propper_img_list, count


def run(args):
    mod = args.img_path_mod
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    print("Coordinate file: " + args.coordinate_file)
    print("Replacement prefix: " + mod)
    print(args)
    completed = 0
    num_reads = 0
    num_skipped = 0
    img_list = None
    if args.image_range:
        img_list = create_img_list(args.image_range)

    start_time = time.time()
    with open(args.coordinate_file, 'r') as co_file:
        with open(args.progress_file, 'a+') as progress_file:
            progress_file.seek(0)

            completed_images = {}
            for line in progress_file:
                line = line.rstrip()
                completed_images[line] = True

            futures = []

            with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
                # Skip the header line
                co_file.readline()

                # Read the first line to initialize the reading func
                line = co_file.readline()
                while co_file.tell() != os.fstat(co_file.fileno()).st_size:
                    rows, filename, line = read_full_image_lines(co_file, mod, line)
                    if filename == "":
                        continue
                    if completed_images.get(filename.split(os.path.sep)[-1]):
                        print("Skipping " + filename)
                        num_skipped += 1
                        continue
                    if img_list is not None:
                        if filename.split("/")[-1].split(".")[0] in img_list:
                            futures.append(executor.submit(process_rows, filename, rows, args))
                            num_reads += 1
                            if num_reads == args.number:
                                break
                        else:
                            num_skipped += 1
                            continue
                    else:
                        futures.append(executor.submit(process_rows, filename, rows, args))
                        num_reads += 1
                        if num_reads == args.number:
                            break

                print("Skipped a total of " + str(num_skipped) + " images")
                for done in cf.as_completed(futures):
                    completed = handle_done(args, completed, done, futures, num_reads, progress_file)
                    if completed == num_reads:
                        break

                print("--- " + str(time.time() - start_time) + " ---")


def handle_done(args, completed, done, futures, num_reads, progress_file):
    rows, fn = done.result()
    fn = fn.split(os.path.sep)[-1]
    progress_file.write(fn + '\n')
    write_field_image(args, fn, rows)
    futures.remove(done)
    completed += 1
    if completed % 10 == 0:
        print(str(completed / num_reads * 100) + "%")
    return completed


def write_field_image(args, fn, rows):
    fn_path = os.path.join(args.output, fn.split(".")[0])
    if not os.path.exists(fn_path):
        os.mkdir(fn_path)
    for row in rows:
        for field in row[0]:
            field_name = os.path.join(fn_path, str(row[1]) + "_" + str(field[1]) + fn)
            cv2.imwrite(field_name, field[0])


def main():
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
    arg_parser.add_argument("--number", "-n", type=int,
                            help="specify the number of images to process starting from the top of the coordinate "
                                 "files or from continuation point specified in the checkpoint file",
                            default=0)
    arg_parser.add_argument("--output", "-o", type=str, help="ouput path", default=".")
    arg_parser.add_argument("--workers", "-w", type=int, help="max number of parallel worker processes", default=4)
    arg_parser.add_argument("--progress_file", type=str,
                            help="location of progress file, will be created if it does not exist",
                            default="progress.txt")
    arg_parser.add_argument("--image_range", "-r", nargs='+',
                            help="Specify a range fsxxxx fsyyyy to process all images in the given range")

    args = arg_parser.parse_args()
    _tmp = []
    for num in args.cols_number:
        _tmp.append(int(num))
    args.cols_number = _tmp
    run(args)


if __name__ == '__main__':
    main()

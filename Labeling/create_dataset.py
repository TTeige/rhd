import os
import argparse
import csv
import cv2


def parse_single_path(in_file_path, out_file_writer, sep, new_base):
    with open(in_file_path, 'r') as f:
        in_reader = csv.DictReader(f)
        for line in in_reader:
            try:
                vals = get_names_labels(line["filename"], line["label"], sep, new_base)
                out_file_writer.writerows(vals)
            except (ValueError, TypeError) as e:
                continue


def get_names_labels(filename, labels, sep, new_base):
    if labels != "" or len(labels) != 3:
        old_full_path = filename.split(sep)
        # The folder name is the same as the image name without the ending
        folder_name = old_full_path[-1].split(".jpg")[0]
        ret_val = []
        for i, l in enumerate(labels):
            single_filename = str(i) + "_" + old_full_path[-1]
            new_full_path = os.path.join(new_base, folder_name, single_filename)
            ret_val.append({"filename": new_full_path, "label": l})
        return ret_val
    else:
        raise ValueError


def parse_single_binary(in_file_path, out_file_writer, sep, new_base):
    with open(in_file_path, 'r') as f:
        in_reader = csv.DictReader(f)
        for line in in_reader:
            try:
                vals = get_names_labels(line["filename"], line["label"], sep, new_base)
                for val in vals:
                    img = cv2.imread(val["filename"])
                    byte_img = img.tostring()
                    val["image"] = byte_img
                out_file_writer.writerows(vals)
            except (ValueError, TypeError, AttributeError) as e:
                if type(e) is AttributeError:
                    print("Exception: {} occured for line {}".format(e, line))
                continue


def parse_all(base_path, out_file_writer, sep, new_base, handle_func):
    for root, subdirs, files in os.walk(base_path):
        for file in files:
            handle_func(os.path.join(root, file), out_file_writer, sep, new_base)


def start(base_path, output_path, input_sep, new_csv_base_path, binary):
    with open(output_path, 'w') as o:
        if binary:
            out_writer = csv.DictWriter(o, ["image", "filename", "label"])
            handle_func = parse_single_binary
        else:
            out_writer = csv.DictWriter(o, ["filename", "label"])
            handle_func = parse_single_path
        out_writer.writeheader()
        parse_all(base_path, out_writer, input_sep, new_csv_base_path, handle_func)


def main():
    arg = argparse.ArgumentParser("Join csv files to a single csv file describing the training set")
    arg.add_argument("-p", "--path", type=str,
                     help="path to root directory containing the csv files.")
    arg.add_argument("-o", "--output", type=str, help="output path of the new csv file")
    arg.add_argument("-s", "--sep", type=str, default="\\", help="separator in the input csv files, assumes all files"
                                                                 "contain the same separator")
    arg.add_argument("--new_base_path", type=str, default="/mnt/remote/Yrke/siffer/",
                     help="base path to add to the training set.")
    arg.add_argument("-b", "--binary", action="store_true", default=False,
                     help="Store the images in the csv file in binary format")
    args = arg.parse_args()
    start(args.path, args.output, args.sep, args.new_base_path, args.binary)


if __name__ == '__main__':
    main()

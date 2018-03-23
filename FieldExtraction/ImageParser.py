import cv2
import numpy as np


class ImageParser:
    def __init__(self, args):
        self.col_names = {"husholdnings_nr": 0,
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
                          "bosatt_i_1946": 25
                          }
        self.cols_number = args.cols_number
        self.cols_name = args.cols_name
        self.type = args.type
        self.process_images = args.process_images
        self.target_fields = []
        if len(self.cols_number) != 0:
            self.target_fields = self.cols_number
        elif len(self.cols_name) != 0:
            self.target_fields = self.cols_name

    def process_rows(self, filename, rows):
        extracted_rows = self._extract_rows(rows)

        image_fields = []
        for i in range(0, len(extracted_rows) - 1, 2):
            row_1 = extracted_rows[i][0]
            row_2 = extracted_rows[i + 1][0]
            fields = self._split_row(filename, row_1, row_2)
            i += 1
            image_fields.append((fields, extracted_rows[i][1]))
        return image_fields, filename

    @staticmethod
    def write_field_image(fn, rows, db):
        for row in rows:
            for field in row[0]:
                field_name = str(row[1]) + "_" + str(field[1]) + fn
                db.store_field(field_name, field[0])

    @staticmethod
    def _convert_img(img):
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
        # for i in range(0, gray_channel.shape[1]):
        #     done = False
        #     for col in gray_channel[:, i]:
        #         if col != 0:
        #             gray_channel = np.delete(gray_channel, np.s_[:i], axis=1)
        #             done = True
        #             break
        #     if done:
        #         break
        # for i in range(gray_channel.shape[1] - 1, 0, -1):
        #     done = False
        #     for col in gray_channel[:, i]:
        #         if col != 0:
        #             gray_channel = np.delete(gray_channel, np.s_[i:], axis=1)
        #             done = True
        #             break
        #     if done:
        #         break

        gray_channel = cv2.bitwise_not(gray_channel)
        #
        # gray_channel = cv2.resize(gray_channel, (60, 60), interpolation=cv2.INTER_AREA)
        #
        # reshaped = np.full((64, 64), 255, dtype='uint8')
        # p = np.array(gray_channel)
        # x_off = y_off = 2
        # reshaped[x_off:p.shape[0] + x_off, y_off:p.shape[1] + y_off] = p

        return gray_channel

    @staticmethod
    def _extract_field(img, row_1, row_2, i):
        # x position different index on same row
        x1 = row_1[i][0]
        x2 = row_1[i + 2][0]
        # y position same index on different row
        y1 = row_1[i][1]
        y2 = row_2[i][1]
        field_img = img[y1:y2, x1:x2]
        return field_img

    def _check_extraction(self, img, row_1, row_2, i):
        field_img = []
        if len(self.target_fields) > 0:
            if row_1[i - 1] in self.target_fields:
                field_img = self._extract_field(img, row_1, row_2, i)
            elif isinstance(self.target_fields[0], str):
                for name in self.target_fields:
                    if self.col_names[name] == row_1[i - 1]:
                        field_img = self._extract_field(img, row_1, row_2, i)

        else:
            field_img = self._extract_field(img, row_1, row_2, i)

        return field_img

    def _split_row(self, img_path, row_1, row_2):
        try:
            fields = []
            img = cv2.imread(img_path)
            if len(img) == 0:
                print("Image not found: " + img_path + " , check path prefix or remote connections")
                return []

            for i in range(1, len(row_1) - 2, 2):
                field_img = self._check_extraction(img, row_1, row_2, i)
                if len(field_img) != 0:
                    if self.process_images:
                        field_img = self._convert_img(field_img)
                    fields.append((field_img, i))
            return fields
        except Exception as e:
            print("Skipping image " + img_path + " Error: ")
            print(e)

    def _extract_rows(self, rows):
        index = 0
        step = 1
        if self.type == "digits":
            step = 2
            index = 1
        elif self.type == "writing":
            step = 2
        extracted_row = []
        for k in range(index, len(rows) - 1, step):
            row_1, row_1_index = self._split_row_str(rows[k])
            row_2, row_2_index = self._split_row_str(rows[k + 1])
            extracted_row.append((row_1, row_1_index))
            extracted_row.append((row_2, row_2_index))

        return extracted_row

    @staticmethod
    def _split_row_str(line):
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

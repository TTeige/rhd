import csv
import shutil
import os

with open("/mnt/remote/Yrke/Yrke_fordelt_fiks.csv") as f:
    base = "/mnt/remote/Yrke/spesifikke_felt/"
    prefix = "/mnt/remote/Yrke/felt/fs1006140"
    reader = csv.DictReader(f)
    for row in reader:
        folder_name = row["1950KODE"]
        if not os.path.exists(base + folder_name):
            os.mkdir(base + folder_name)
        if not row["1950KODE"]:
            folder_name = "uspesifikk"
            if not os.path.exists(base + folder_name):
                os.mkdir(base + folder_name)
        try:
            filename = prefix + row["BILDEID"] + "/" + str(int(row["RAD"]) + 2) + "_27fs1006140" + row["BILDEID"] + ".jpg"
            cp_filename = base + folder_name + "/" + str(int(row["RAD"]) + 2) + "_27fs1006140" + row["BILDEID"] + ".jpg"
            if not os.path.exists(cp_filename):
                try:
                    shutil.copy(filename, cp_filename)
                except OSError as e:
                    print(e)
        except ValueError as e:
            print(row)

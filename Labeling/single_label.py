# import gi
import os
import csv
# gi.require_version('Gtk', '3.0')
# from gi.repository import Gtk, Gdk
# from gi.repository import GdkPixbuf

import tkinter as tk
from tkinter import filedialog, StringVar
from PIL import Image, ImageTk


class SingleView(tk.Frame):
    def __init__(self, master=None):
        if not os.path.exists("labels.csv"):
            with open("labels.csv", "w") as l:
                writer = csv.DictWriter(l, ["filename", "label"])
                writer.writeheader()
        super().__init__(master)
        self.pack()
        self.img_dir = ""
        self.root_dir = ""
        self.entry_text = StringVar()

        self.current_files = []
        self.available_dirs = []
        self.img_index = 0
        self.folder_index = 0

        self.progress = self.restore_from_file()

        self.welcome_label = tk.Label(self,
                                      text="Click Choose Directory and select the directory containing the images to be labeled")

        self.img = tk.Label(self, image=None)

        self.entry = tk.Entry(self, textvariable=self.entry_text)
        self.entry.bind("<Return>", self.on_submit_entry)

        self.select_folder_btn = tk.Button(self, command=self.on_folder_clicked, text="Select Folder")

        prev_btn = tk.Button(self, command=self.prev_img, text="Previous Image")

        submit = tk.Button(self, command=self.on_submit, text="Submit")
        submit.bind("<Return>", self.on_submit)

        self.grid(column=0, row=0, columnspan=3, rowspan=6)
        self.welcome_label.grid(column=0, row=0, columnspan=3, rowspan=1)
        self.img.grid(column=0, row=1, columnspan=3, rowspan=2)
        self.entry.grid(column=0, row=4, columnspan=3, rowspan=1)
        self.select_folder_btn.grid(column=0, row=5, columnspan=1, rowspan=1)
        prev_btn.grid(column=1, row=5, columnspan=1, rowspan=1)
        submit.grid(column=2, row=5, columnspan=1, rowspan=1)

    def prev_img(self):
        self.img_index -= 2
        if self.img_index < 0:
            self.img_index = 0
            self.folder_index -= 2
            if self.folder_index < 0:
                self.folder_index = 0
                return
            self.move_to_next_folder()
        self.update_img()
        with open("labels.csv", "a+") as csv_file:
            # Taken from https://stackoverflow.com/a/10289740
            # Move the pointer (similar to a cursor in a text editor) to the end of the file.
            csv_file.seek(0, os.SEEK_END)

            # This code means the following code skips the very last character in the file -
            # i.e. in the case the last line is null we delete the last line
            # and the penultimate one
            pos = csv_file.tell() - 1

            # Read each character in the file one at a time from the penultimate
            # character going backwards, searching for a newline character
            # If we find a new line, exit the search
            while pos > 0 and csv_file.read(1) != "\n":
                pos -= 1
                csv_file.seek(pos, os.SEEK_SET)

            # So long as we're not at the start of the file, delete all the characters ahead of this position
            if pos > 0:
                # Skips the \n character
                pos += 1
                # Truncate the file after \n
                csv_file.seek(pos, os.SEEK_SET)
                csv_file.truncate()

    @staticmethod
    def restore_from_file():
        done_dict = {}
        with open("labels.csv", "r") as prog:
            reader = csv.DictReader(prog)
            for row in reader:
                done_dict[row["filename"]] = True
                folder = row["filename"]
                folder = folder.split(os.path.sep)[:-1]
                folder = os.path.sep.join(folder)
                done_dict[folder] = True
        return done_dict

    def on_submit_entry(self, a):
        self.on_submit()

    def on_submit(self):
        with open("labels.csv", "a+") as labels:
            writer = csv.DictWriter(labels, ["filename", "label"])

            try:
                writer.writerow({"filename": self.current_files[self.img_index], "label": self.entry_text.get()})
                self.entry_text.set("")
                self.update_img()
            except IndexError as e:
                self.move_to_next_folder()

    def update_img(self):
        self.img_index += 1
        path = self.current_files[self.img_index]
        if path not in self.progress:
            try:
                tmp_img = ImageTk.PhotoImage(Image.open(path))
                self.img.configure(image=tmp_img)
                self.img.image = tmp_img
                self.welcome_label["text"] = path
            except Exception as e:
                print(e)
        else:
            self.update_img()

    def select_files(self, folder):
        self.current_files = [os.path.join(folder, f) for f in os.listdir(folder) if
                              os.path.isfile(os.path.join(folder, f))]

    def get_dirs(self, folder):
        self.available_dirs = [os.path.join(folder, f) for f in os.listdir(folder) if
                               os.path.isdir(os.path.join(folder, f))]

    def on_folder_clicked(self):
        self.img_dir = filedialog.askdirectory()
        if self.img_dir == "":
            return
        # Have to replace all of the "/" because tkinter only returns a path containing them
        components = self.img_dir.split("/")
        if os.path.sep == "\\":
            components[0] += "\\"
        self.img_dir = ""
        for i in components:
            self.img_dir = os.path.join(self.img_dir, i)
        try:
            self.get_dirs(self.img_dir)
            self.select_files(self.img_dir)
            self.update_img()
        except IndexError as e:
            self.move_to_next_folder()

    def move_to_next_folder(self):
        self.folder_index += 1
        try:
            self.img_dir = self.available_dirs[self.folder_index]
            if self.img_dir not in self.progress:
                self.select_files(self.img_dir)
                self.img_index = 0
                self.update_img()
            else:
                self.move_to_next_folder()
        except IndexError:
            self.welcome_label[
                "text"] = "Click submit again, if there are no new images, you are done with the selected directory!"


if __name__ == '__main__':
    root = tk.Tk()
    window = SingleView(master=root)
    window.mainloop()

import gi
import os
import csv

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from gi.repository import GdkPixbuf


class SingleView(Gtk.Window):
    def __init__(self):
        if not os.path.exists("labels.csv"):
            with open("labels.csv", "w") as l:
                writer = csv.DictWriter(l, ["filename", "label"])
                writer.writeheader()
        Gtk.Window.__init__(self, title="Simple Image Labeling")
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(main_box)
        self.img_dir = ""
        self.root_dir = ""

        self.current_files = []
        self.available_dirs = []
        self.img_index = 0
        self.folder_index = 0

        self.progress = self.restore_from_file()

        self.welcome_label = Gtk.Label(
            "Click Choose Directory and select the directory containing the images to be labeled")
        main_box.add(self.welcome_label)

        self.img = Gtk.Image()
        main_box.add(self.img)

        self.entry = Gtk.Entry()
        self.entry.connect("activate", self.on_submit)
        main_box.add(self.entry)

        button_container = Gtk.Box(spacing=6)

        select_folder_btn = Gtk.Button("Choose Directory")
        select_folder_btn.connect("clicked", self.on_folder_clicked)
        button_container.pack_start(select_folder_btn, True, True, 0)

        prev_btn = Gtk.Button("Previous Image")
        prev_btn.connect("clicked", self.prev_img)
        button_container.pack_start(prev_btn, True, True, 0)

        submit = Gtk.Button.new_with_label("Submit")
        submit.connect("clicked", self.on_submit)
        button_container.pack_start(submit, True, True, 0)

        main_box.add(button_container)

    def prev_img(self, btn):
        self.img_index -= 2
        if self.img_index < 0:
            self.img_index = 0
            self.folder_index -= 2
            if self.folder_index < 0:
                self.folder_index = 0
                return
            self.move_to_next_folder()
        self.update_img()
        self.img.show()
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

    def on_submit(self, btn):
        with open("labels.csv", "a+") as labels:
            writer = csv.DictWriter(labels, ["filename", "label"])

            text = self.entry.get_text()

            if text == "":
                text = "None"
            try:
                writer.writerow({"filename": self.current_files[self.img_index], "label": text})

                self.entry.set_text("")
                self.update_img()
            except IndexError as e:
                self.move_to_next_folder()

    def update_img(self):
        self.img_index += 1
        path = self.current_files[self.img_index]
        if path not in self.progress:
            self.img.set_from_pixbuf(GdkPixbuf.Pixbuf.new_from_file(path))
            self.welcome_label.set_label(path)
        else:
            self.update_img()

    def select_files(self, folder):
        self.current_files = [os.path.join(folder, f) for f in os.listdir(folder) if
                              os.path.isfile(os.path.join(folder, f))]

    def get_dirs(self, folder):
        self.available_dirs = [os.path.join(folder, f) for f in os.listdir(folder) if
                               os.path.isdir(os.path.join(folder, f))]

    def on_folder_clicked(self, btn):
        dialog = Gtk.FileChooserDialog("Select folder containing images", self, Gtk.FileChooserAction.SELECT_FOLDER
                                       , (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                          "Select", Gtk.ResponseType.OK))
        dialog.set_default_size(800, 400)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.img_dir = dialog.get_filename()
            try:
                self.get_dirs(self.img_dir)
                self.select_files(self.img_dir)
                self.update_img()
            except IndexError as e:
                self.move_to_next_folder()

        dialog.destroy()

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
            self.welcome_label.set_label(
                "Click submit again, if there are no new images, you are done with the selected directory!")


if __name__ == '__main__':
    window = SingleView()
    window.connect('delete-event', Gtk.main_quit)
    window.show_all()
    Gtk.main()

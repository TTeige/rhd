import gi
import os
import sys
import csv

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from gi.repository import GdkPixbuf

# Dirty workaround for checking if the labels file exist
file_exists = os.path.isfile("labels.csv")


class MainWindow(Gtk.Window):
    def __init__(self, image_path, csv_file):

        self.image_paths_set = set()
        self.init_image_paths(image_path)
        self.image_index = 0
        self.labeled_image_count = 0
        self.current_file = ""

        self.csv_file = csv_file
        fieldnames = ['filename', 'label', 'resolved']

        self.resume(fieldnames, csv_file)

        self.image_paths = list(self.image_paths_set)
        self.num_images = len(self.image_paths)

        self.writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            self.writer.writeheader()

        Gtk.Window.__init__(self, title='Labeling')
        self.set_size_request(1028, 800)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # filename_box = Gtk.Box(spacing=6)
        # self.filename_label = Gtk.Label(str.format("{}", self.current_file))
        # filename_box.pack_start(self.filename_label, True, True, 0)
        # vbox.pack_start(filename_box, True, True, 0)

        next_prev_box = Gtk.Box(spacing=6)
        prev_image = Gtk.Button.new_with_label("Prev")
        prev_image.connect("clicked", self.prev_image)
        next_prev_box.pack_start(prev_image, True, True, 0)
        next_image = Gtk.Button.new_with_label("Next")
        next_image.connect("clicked", self.next_image)
        next_prev_box.pack_start(next_image, True, True, 0)
        # self.img_count = Gtk.Label(str.format("{}/{}", self.labeled_image_count, self.num_images))
        # next_prev_box.pack_start(self.img_count, True, True, 0)
        vbox.pack_start(next_prev_box, True, True, 0)

        hbox = Gtk.Box(spacing=6)
        vbox.pack_start(hbox, True, True, 0)

        # self.entry = Gtk.Entry()
        # self.entry.set_text("Enter label")
        # self.entry.connect("activate", self.on_submit_clicked)
        # hbox.pack_start(self.entry, True, True, 0)
        # submit = Gtk.Button.new_with_label("Submit")
        # submit.connect("clicked", self.on_submit_clicked)
        # hbox.pack_start(submit, True, True, 0)

        self.images = []
        for i in range(self.image_index, self.image_index + 3):
            img = Gtk.Image()
            img.set_from_pixbuf(GdkPixbuf.Pixbuf.new_from_file(self.image_paths[i]))
            self.images.append(img)
            self.image_index += 1
            vbox.pack_start(self.images[i], True, True, 0)

    def do_submit(self):
        label = self.entry.get_text()
        print(label)
        self.entry.set_text("")
        self.go_to_next()
        if label == "":
            label = "NaN"
        self.writer.writerow({'filename': self.image_paths[self.image_index], 'label': label})
        self.labeled_image_count += 1
        self.img_count.set_text(str.format("{}/{}", self.labeled_image_count, self.num_images))

    def on_submit_clicked(self, button):
        self.do_submit()

    def go_to_next(self):
        self.images = []
        for i in range(self.image_index, self.image_index + 3):
            img = Gtk.Image()
            img.set_from_pixbuf(GdkPixbuf.Pixbuf.new_from_file(self.image_paths[i]))
            self.image_index += 1
            self.images[i] = img

    def next_image(self, button):
        self.go_to_next()

    def prev_image(self, button):
        self.image_index -= 1
        self.display_image = GdkPixbuf.Pixbuf.new_from_file(self.image_paths[self.image_index])
        self.img.set_from_pixbuf(self.display_image)

        # Taken from https://stackoverflow.com/a/10289740
        # Move the pointer (similar to a cursor in a text editor) to the end of the file.
        self.csv_file.seek(0, os.SEEK_END)

        # This code means the following code skips the very last character in the file -
        # i.e. in the case the last line is null we delete the last line
        # and the penultimate one
        pos = self.csv_file.tell() - 1

        # Read each character in the file one at a time from the penultimate
        # character going backwards, searching for a newline character
        # If we find a new line, exit the search
        while pos > 0 and self.csv_file.read(1) != "\n":
            pos -= 1
            self.csv_file.seek(pos, os.SEEK_SET)

        # So long as we're not at the start of the file, delete all the characters ahead of this position
        if pos > 0:
            # Skips the \n character
            pos += 1
            # Truncate the file after \n
            self.csv_file.seek(pos, os.SEEK_SET)
            self.csv_file.truncate()

    def init_image_paths(self, images):
        for dir, dirs, fns in os.walk(images):
            for fn in fns:
                if fn.split('.')[1] != "jpg":
                    continue
                fp = os.path.join(dir, fn)
                self.image_paths_set.add(fp)

    def resume(self, fieldnames, csv_file):
        if file_exists:
            labeled_images = set()
            reader = csv.reader(csv_file)
            for row in reader:
                labeled_images.add(row[0])

            self.image_paths_set = self.image_paths_set - labeled_images


def main(argv):
    if len(argv) < 2:
        print("Usage: python3 simple_labeling.py <path/to/images>")
        return

    with open('labels.csv', 'a+') as csv_file:
        csv_file.seek(0)
        window = MainWindow(argv[1], csv_file)
        window.connect('delete-event', Gtk.main_quit)
        window.show_all()
        Gtk.main()


if __name__ == '__main__':
    main(sys.argv)

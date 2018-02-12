import gi
import sys
import os
import csv

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from gi.repository import GdkPixbuf


class SingleImage:

    def __init__(self, image_path):
        self.path = image_path
        self.img = Gtk.Image()
        self.set_image(self.path)
        self.checked = False
        self.check_button = Gtk.ToggleButton()
        self.check_button.connect("toggled", self.on_click)
        self.check_button.add(self.img)

    def set_image(self, path):
        self.path = path
        display_image = GdkPixbuf.Pixbuf.new_from_file_at_scale(self.path, 100, 100, preserve_aspect_ratio=True)
        self.img.set_from_pixbuf(display_image)

    def on_click(self, button):
        self.checked = button.get_active()


class DirectoriesClusteredView(Gtk.Window):

    def __init__(self, images_path, csv):
        self.root = images_path
        self.progress = csv
        self.labels = ["filename", "label"]

        self.directories = self.get_directories(self.root)
        self.dir_index = 0

        Gtk.Window.__init__(self, title='Labeling')
        self.set_size_request(400, 200)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        self.img_index = 0
        self.current_dir_label = Gtk.Label(self.directories[self.dir_index].split(os.path.sep)[-2])
        vbox.pack_start(self.current_dir_label, True, True, 0)

        self.images_box, self.images = self.load_flowbox()
        vbox.pack_start(self.images_box, True, True, 0)

        next_prev_box = Gtk.Box(spacing=6)
        prev_image = Gtk.Button.new_with_label("Prev")
        prev_image.connect("clicked", self.next_folder, -1)
        next_prev_box.pack_start(prev_image, True, True, 0)
        next_image = Gtk.Button.new_with_label("Next")
        next_image.connect("clicked", self.next_folder, 1)
        next_prev_box.pack_start(next_image, True, True, 0)
        submit = Gtk.Button.new_with_label("Submit")
        submit.connect("clicked", self.submit)
        next_prev_box.pack_start(submit, True, True, 0)
        vbox.pack_start(next_prev_box, True, True, 0)

    def submit(self, btn):
        for img in self.images:
            if img.checked:
                print(img.path)

    def next_folder(self, btn, direction):
        self.dir_index += direction
        if self.dir_index >= len(self.directories):
            return
        self.current_dir_label = Gtk.Label(self.directories[self.dir_index])
        self.images_box, self.images = self.load_flowbox()

    @staticmethod
    def get_directories(path):
        return [os.path.join(path, x[0]) for x in os.walk(path)]

    @staticmethod
    def get_files(path):
        return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    def load_flowbox(self):
        images = []
        images_box = Gtk.FlowBox()
        images_box.set_valign(Gtk.Align.START)
        images_box.set_max_children_per_line(30)
        images_box.set_selection_mode(Gtk.SelectionMode.NONE)
        files = self.get_files(self.directories[self.dir_index])
        for f in files:
            img_box = SingleImage(f)
            images_box.add(img_box.check_button)
            images.append(img_box)

        return images_box, images


class LabelFileClusterView(Gtk.Window):
    def __init__(self, progress_file, label_file):
        Gtk.Window.__init__(self, title='Labeling')
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(main_box)
        self.init_boxes(main_box)
        unique_labels = self.init_labels(label_file)
        self.init_selector(unique_labels, main_box)

    def init_selector(self, labels, main_box):
        label_store = Gtk.ListStore(str)
        for l in labels:
            label_store.append([l])

        label_combo = Gtk.ComboBox.new_with_model(label_store)
        label_combo.connect("changed", self.on_label_combo_changed)
        renderer_text = Gtk.CellRendererText()
        label_combo.pack_start(renderer_text, True)
        label_combo.add_attribute(renderer_text, "text", 0)
        main_box.pack_start(label_combo, False, False, True)

    def on_label_combo_changed(self, combo):
        tree_iter = combo.get_active_iter()
        if tree_iter is not None:
            model = combo.get_model()
            label = model[tree_iter][0]
            print(label)

    @staticmethod
    def init_labels(label_file):
        reader = csv.DictReader(label_file)
        label_set = set()
        for entry in reader:
            label_set.add(entry['kode'])
        return label_set

    def next_batch(self, btn, direction):
        pass

    def init_boxes(self, main_box):
        next_prev_box = Gtk.Box(spacing=6)
        # prev_image = Gtk.Button.new_with_label("Prev")
        # prev_image.connect("clicked", self.next_batch, -1)
        # next_prev_box.pack_start(prev_image, True, True, 0)
        # next_image = Gtk.Button.new_with_label("Next")
        # next_image.connect("clicked", self.next_batch, 1)
        # next_prev_box.pack_start(next_image, True, True, 0)
        # submit = Gtk.Button.new_with_label("Submit")
        # submit.connect("clicked", self.submit)
        # next_prev_box.pack_start(submit, True, True, 0)
        # main_box.pack_start(next_prev_box, True, True, 0)

def main(argv):
    with open('progress.csv', 'a+') as progress_file:
        progress_file.seek(0)

        if argv[2] != "file":

            window = DirectoriesClusteredView(argv[1], progress_file)
            window.connect('delete-event', Gtk.main_quit)
            window.show_all()
            Gtk.main()

        else:
            with open(sys.argv[1], 'r') as label_file:
                window = LabelFileClusterView(progress_file, label_file)
                window.connect('delete-event', Gtk.main_quit)
                window.show_all()
                Gtk.main()

if __name__ == '__main__':
    main(sys.argv)

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

        self.label_file = label_file
        self.index = 0

        self.selected_boxes = []
        self.count = Gtk.Label("0/0")
        Gtk.Window.__init__(self, title='Labeling')
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(main_box)
        unique_labels, self.img_paths = self.init_labels(label_file)
        self.init_selector(unique_labels, main_box)
        self.current_label = ""

        self.grid = Gtk.Grid()
        self.widgets = []
        toggle_button = self.create_toggle_button(0)
        self.widgets.append(toggle_button)
        self.grid.add(toggle_button)
        prev_height = toggle_button
        prev_button = toggle_button
        btn_index = 1
        for i in range(0, 4):
            toggle_button = self.create_toggle_button(btn_index)
            btn_index += 1
            self.widgets.append(toggle_button)
            self.grid.attach_next_to(toggle_button, prev_height, Gtk.PositionType.BOTTOM, 1, 1)
            prev_height = toggle_button

        for i in range(0, 5):
            toggle_button = self.create_toggle_button(btn_index)
            btn_index += 1
            self.widgets.append(toggle_button)
            self.grid.attach_next_to(toggle_button, prev_button, Gtk.PositionType.RIGHT, 1, 1)
            prev_button = toggle_button
            prev_height = prev_button
            for k in range(0, 4):
                toggle_button = self.create_toggle_button(btn_index)
                btn_index += 1
                self.widgets.append(toggle_button)
                self.grid.attach_next_to(toggle_button, prev_height, Gtk.PositionType.BOTTOM, 1, 1)
                prev_height = toggle_button

        self.image_ids = []
        main_box.pack_start(self.grid, True, True, False)
        self.init_boxes(main_box)
        self.show_all()

    def create_toggle_button(self, btn_index):
        toggle = Gtk.ToggleButton()
        toggle.set_size_request(150, 50)
        toggle.connect("toggled", self.on_click, btn_index)
        toggle.add(Gtk.Image())
        return toggle

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
            self.current_label = label
            self.index = 0
            self.load_images(self.img_paths[self.current_label], self.widgets, self.index)

    def load_images(self, files, widgets, index):
        for i in self.widgets:
            i.get_child().clear()
            i.set_active(False)
        widget_index = 0
        for i in range(index, len(files)):
            if widget_index >= len(widgets):
                break
            if os.path.isfile(files[i]):
                display_image = GdkPixbuf.Pixbuf.new_from_file_at_scale(files[i], 100, 100, preserve_aspect_ratio=True)
                widgets[widget_index].get_child().set_from_pixbuf(display_image)
                widget_index += 1
        total = int(len(files) / len(self.widgets) + 1)
        cur = int(self.index / len(files))
        self.count.set_text("{}/{}".format(cur, total))

    def on_click(self, btn, index):
        if btn.get_active():
            self.selected_boxes.append(index)
        else:
            self.selected_boxes.remove(index)

    @staticmethod
    def init_labels(label_file):
        reader = csv.DictReader(label_file)
        label_set = set()
        paths = {}
        for entry in reader:
            label_set.add(entry['1950KODE'])
            try:
                if entry['1950KODE'] in paths:
                    paths[entry['1950KODE']].append(
                        os.path.join("/mnt", "remote", "Yrke", "felt", "fs1006140" + entry['BILDEID'],
                                     str(int(entry['RAD']) + 2) + "_27" + "fs1006140" + entry['BILDEID'] + ".jpg"))
                else:
                    paths[entry['1950KODE']] = [
                        os.path.join("/mnt", "remote", "Yrke", "felt", "fs1006140" + entry['BILDEID'],
                                     str(int(entry['RAD']) + 2) + "_27" + "fs1006140" + entry['BILDEID'] + ".jpg")]
            except Exception as e:
                print(e)
        removed = []
        with open("wrong_code.csv") as wrong:
            reader = csv.DictReader(wrong)
            for row in reader:
                removed.append((row["filename"], row["code"]))

        for i in removed:
            paths[i[1]].remove(i[0])
        for k, v in paths.items():
            if len(v) == 0:
                label_set.remove(k)

        return sorted(list(label_set)), paths

    def next_batch(self, btn, direction):

        files = self.img_paths[self.current_label]

        if self.index + direction * 30 > len(files):
            return

        self.index += direction * 30

        if self.index < 0:
            self.index = 0

        self.load_images(files, self.widgets, self.index)

    def submit(self, btn):
        files = self.img_paths[self.current_label]

        if len(self.selected_boxes) > 0:
            selected_files = []
            for i in self.selected_boxes:
                selected_files.append(files[self.index + i])
            with open("wrong_code.csv", 'a+') as csv_file:
                self.store_selection(csv_file, selected_files)
            self.load_images(files, self.widgets, self.index)

    def store_selection(self, csv_file, files):
        writer = csv.DictWriter(csv_file, ["filename", "code"])
        for f in files:
            writer.writerow({"filename": f, "code": self.current_label})
            self.img_paths[self.current_label].remove(f)

    def init_boxes(self, main_box):
        next_prev_box = Gtk.Box(spacing=6)
        prev_image = Gtk.Button.new_with_label("Prev")
        prev_image.connect("clicked", self.next_batch, -1)
        next_prev_box.pack_start(prev_image, True, True, 0)
        next_image = Gtk.Button.new_with_label("Next")
        next_image.connect("clicked", self.next_batch, 1)
        next_prev_box.pack_start(next_image, True, True, 0)
        submit = Gtk.Button.new_with_label("Submit")
        submit.connect("clicked", self.submit)
        next_prev_box.pack_start(submit, True, True, 0)
        next_prev_box.pack_start(self.count, True, True, 0)
        main_box.pack_start(next_prev_box, True, True, 0)


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

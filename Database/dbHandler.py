import sqlite3


class DbHandler:
    def __init__(self, db_loc, connection=None):
        self.db_loc = db_loc
        self.connection = connection
        self.cursor = None
        if self.connection is None:
            self.connection = sqlite3.connect(db_loc)

    def store_field(self, name, img):
        # Name, raw byte data, height width
        self.connection.cursor().execute('insert or IGNORE into fields VALUES (?, ?, ?, ?)', (name, img, img.shape[0], img.shape[1]))

    def store_digit(self, name, img):
        self.connection.cursor().execute('insert or IGNORE into digit VALUES (?, ?)', (name, img))

    def store_dropped(self, name, reason):
        self.connection.cursor().execute('insert or IGNORE into dropped VALUES (?, ?)', (name, reason))

    def test_exists(self, name):
        c = self.connection.cursor()
        c.execute("SELECT EXISTS(SELECT 1 FROM fields WHERE name LIKE :name LIMIT 1)", {'name': name})
        return c.fetchone()

    def test_exists_digit(self, name):
        c = self.connection.cursor()
        c.execute("SELECT EXISTS(SELECT 1 FROM digit WHERE name LIKE :name LIMIT 1)", {'name': "%"+name+"%"})
        return c.fetchone()

    def test_exists_dropped(self, name):
        c = self.connection.cursor()
        c.execute("SELECT EXISTS(SELECT 1 FROM dropped WHERE name LIKE :name LIMIT 1)", {'name': "%" + name + "%"})
        return c.fetchone()

    def select_image(self, name):
        c = self.connection.cursor()
        c.execute("SELECT * FROM fields WHERE name=:name", {'name': name})
        return c.fetchone()

    def select_all_images(self):
        return self.connection.cursor().execute("SELECT * FROM fields")

    def count_rows_in_fields(self):
        return self.connection.cursor().execute("SELECT Count(*) FROM fields")

    def count_rows_in_digit(self):
        return self.connection.cursor().execute("SELECT Count(*) FROM digit")

    def count_rows_in_dropped(self):
        return self.connection.cursor().execute("SELECT Count(*) FROM dropped")

    def __enter__(self):
        try:
            if self.connection is None:
                self.connection = sqlite3.connect(self.db_loc)
            return self
        except Exception as e:
            print(e)
            exit(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.commit()
        self.connection.close()

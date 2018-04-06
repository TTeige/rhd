import sqlite3

conn = sqlite3.connect("ft1950_ml.db")
c = conn.cursor()
c.execute('''CREATE TABLE fields (name varchar PRIMARY KEY, image blob, height int, width int)''')
c.execute('''CREATE TABLE dropped (name varchar PRIMARY KEY, reason text)''')
c.execute('''CREATE TABLE digit (name varchar PRIMARY KEY, image blob)''')
c.connection.commit()
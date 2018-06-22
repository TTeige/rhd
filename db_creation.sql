CREATE TABLE digit
(
  name  VARCHAR
    PRIMARY KEY,
  image BLOB
);

CREATE TABLE dropped
(
  name   VARCHAR
    PRIMARY KEY,
  reason TEXT
);

CREATE TABLE fields
(
  name   VARCHAR
    PRIMARY KEY,
  image  BLOB,
  width  INT,
  height INT
);

CREATE TABLE tmp_fields
(
  name   VARCHAR
    PRIMARY KEY,
  image  BLOB,
  width  INT,
  height INT
);


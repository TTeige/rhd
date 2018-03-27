# RHD

This repository contains files in regards to work done for Registreringssentralen fro historisk data. The code is intended for usage on the 1950 consensus in Norway.

## Naming convetion of images in the database

x_y_zfs10....jpg. 
- __x__ - Identifies the index in the field
- __y__ - Identifies the row index in the image
- __z__ - Identifies the field index in the image

## Image Parser

The image parser directory contains a program for parsing 1950s concencus images based on a coordinate file provided by AnalyseForm.

The program has the following dependencies:
- OpenCV
- Numpy
- python 3.x

These can be installed by using pip

```bash
pip install numpy
pip install opencv
```
### Usage

This command can be used for only extracting the fields of all the data, default output directory is ".".
```
python3 extract_fields.py <path/to/coordinate_file>
```

Use `python3 FieldSplitter.py help` for more parameters

### Program Walkthrough
- __extract_fields.py__ - Parses the input arguments and runs FieldSplitter.run()
- __FieldSplitter.py__ - Initializes CoordinateFileReader, ProgressFileHandler, ImageParser, DbHandler and the ParallelExecutor.
Launches the ParallelExecutor.run() method
- __ParallelExecutor.py__ - Gets a list of images from the CoordinateFileReader. Creates a pool of processes which are launched 
with the ImageParser.process_rows() method. Writes the images from the ImageParser to the database using the DbHandler. Continiously uses the 
ProgressFileHandler to write the progress.
- __CoordinateFileReader.py__ - Process the input coordinate file and creates a list of filepaths containing images 
to be processed.
- __ProgressFileHandler.py__ - Writes the progress.
- __DbHandler.py__ - Handles the interaction between ParallelExecutor and the database. 
- __ImageParser.py__ - Extracts single fields from entire census images.
  - ProcessRows - Extracts each row based on input argument. It can extract the digit rows, the text rows or both. Extracts two rows at a time, since the coordinate file defines only the top value of the row. So the next row has to be read at the same time.
    - _split_row_string - Extracts the coordinates from the coordinates file in field format.
    - _split_row - Extracts all the fields from the given row. The images are loaded in this method. This should be changed, since it leads to the image being loaded more than a single time. 
      - _check_extraction - Verifies that the field is a valid field, based on the list of given fields that are to be extracted.
        - _extract_field - Fetches the given segment from the image. 
      - _convert_image - Converts the image to grayscale and mask the digit from the field.

### Notes
Can contain bugs when it comes to defining target columns. See the constructor of ImageParser.py. 

## Clustering
The clustering directory contains the script for using the multi component gaussian normal distribution clustering algorithm. The directory also contains the algorithm for clustering the training set based on the geometric shapes in the single digit images.

### Notes
Currently the multi component gaussian distribution clustering algorithm only support three digit images. There is a simple solution to fix this. Depending on the image name the expected number of local maxima and minimas should change. 

## Machine Learning - Handwritten digit training and predictions

The machine learning directory contains a program which can either predict images or train a model. The current script creates a model which is not useable on the concensus data. This will be fixed.

It also contains a script which generates 3 digit images and labels using the MNIST dataset. This did not work for predicting the concencus images.

Dependencies:

- Tensorflow
- Numpy
- OpenCV
- python 3.x

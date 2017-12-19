# RHD

This repository contains files in regards to work done for Registreringssentralen fro historisk data. The code is intended for usage on the 1950 consensus in Norway.

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
python3 FieldSplitter.py <path/to/coordinate_file>
```

Use `python3 FieldSplitter.py help` for more parameters


## Machine Learning - Handwritten digit training and predictions

The machine learning directory contains a program which can either predict images or train a model. The current script creates a model which is not useable on the concensus data. This will be fixed.

It also contains a script which generates 3 digit images and labels using the MNIST dataset. This did not work for predicting the concencus images.

Dependencies:

- Tensorflow
- Numpy
- OpenCV
- python 3.x

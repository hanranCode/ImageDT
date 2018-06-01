# ImageDT
a python lib for neural networks, file and image processing etc.

Installation
------------
pip install imagedt

Dependencies
------------
## Install coremltools 
pip install coremltools


*coremltools* has the following dependencies:

- numpy (1.10.0+)
- protobuf (3.1.0+)

In addition, it has the following soft dependencies that are only needed when
you are converting models of these formats:

- Keras (1.2.2, 2.0.4+) with corresponding Tensorflow version
- Xgboost (0.7+)
- scikit-learn (0.17+)
- libSVM
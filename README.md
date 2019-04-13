# ImageDT
<font size=5 > a python lib for neural networks, file and image processing etc. </font>

Installation
------------
<font size=5 >1.  pip install
    pip install imagedt <br>
2. source install </font>

    #1  pip install imagedt
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #2  git clone https://github.com/hanranCode/ImageDT.git
        cd ImageDT 
        python setup.py install
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #3  vi ~/.bashrc
        export PYTHONPATH="/path_to_ImageDT:$PYTHONPATH"
        import imagedt in your project


Dependencies
------------
<font size=4 >1. tensorflow <br>
2. coremltools <br>
*Coremltools* has the following dependencies:

- numpy (1.10.0+)
- protobuf (3.1.0+)

In addition, it has the following soft dependencies that are only needed when
you are converting models of these formats:

- Keras (1.2.2, 2.0.4+) with corresponding Tensorflow version
- Xgboost (0.7+)
- scikit-learn (0.17+)
- libSVM  <br>
</font><br>


TODO
------------
- <font size=6 > <b>tools </b></font> <br>
    - [x] Converte VOC data to tfrecords <br>
    - [ ] Data interface  <br>
- <font size=6 ><b> networks </b></font><br>

    - <font size=5 >tensorflow</font>
        + [x] lenet test
        + [ ] ssd_mobilenet <br>

    - <font size=5 >caffe</font>
        - [ ] <font size=4 >? </font> <br>

    - <font size=5 >pyroch</font>
        - [ ] ?? <br>
        
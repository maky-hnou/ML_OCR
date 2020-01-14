# ML_OCR  

![Python version][python-version]
[![GitHub issues][issues-image]][issues-url]
[![GitHub forks][fork-image]][fork-url]
[![GitHub Stars][stars-image]][stars-url]
[![License][license-image]][license-url]

## About this repo:  

This repository is a re-implementation of an OCR using Machine Learning.  
It has been developed on Ubuntu 18.04 using python 3.6 and TensorFlow 1.13.   
The original implementation is available via this [link](https://github.com/vinayakkailas/Deeplearning-OCR).


## Content:  

- **cnn_model:** a folder containing the needed files to build the CRNN.  
- **data:** a folder containing the char dicts.
- **data_provider:** a folder containing the needed file to prepare and provide the data to the CRNN.
- **global_configuration:** a folder containing the needed configuration for the CRNN.  
- **local_utils:** a folder containing the utils needed to manipulate the data, char dicts and to write the logs.  
- **tools:** a folder containing the needed files to generate the tfrecords, train the CRNN and to test it.  
- **generate_tfrecords.py:** the file used to generate tfrecords.  
- **test_ocr.py:** the file used to test the OCR.
- **train_ocr.py:** the file used to train the CRNN.  
- **requirements.txt:** a text file containing the needed packages to run the project (if you want to use GPU instead of CPU, change tensorflow to tensorflow-gpu in requirements.txt).  


## Train and test the CRNN:  

**1. Prepare the environment:**  
*NB: Use python 3+ only.*  
Before anything, please install the requirements by running: `pip install -r requirements.txt`.  

**2. Prepare the data:**  
Download the synth90k dataset available via this [link](http://www.robots.ox.ac.uk/~vgg/data/text/) or you can use your own data.  
Extract all the files into a `imgs/` directory.  
The extracted data into `imgs/` should be organized as follows:  
`imgs` should contain two folders named `Train` and `Test` each one contains images and a text file named `sample.txt` that contains the image name and its label separated by a blank space according the this format 'imagename image label' example: (img1.jpg hello)  
Convert the whole dataset into tensorflow records by running `python generate_tfrecords.py`.  
After finishing, 3 tfrecods will be saved to `tfrecs/`: one for training, one for validating and the other for testing the model.   

**3. Train the CRNN:**  
To train the shadownet, run `python train_ocr.py`.   
The trained model will be saved to a directory named `ML_OCR/model/shadownet/`.  

**4. Test the model:**  
To test the model, images test should be in tfrecord form (After generating tfrecords, you should have `test.tfrecord` file in `tfrec/`).
Add the model name to the path in `test_ocr.py`.    
You can test the model by running `python test_ocr.py`.  

[python-version]:https://img.shields.io/badge/python-3.6+-brightgreen.svg
[issues-image]:https://img.shields.io/github/issues/maky-hnou/ML_OCR.svg
[issues-url]:https://github.com/maky-hnou/ML_OCR/issues
[fork-image]:https://img.shields.io/github/forks/maky-hnou/ML_OCR.svg
[fork-url]:https://github.com/maky-hnou/ML_OCR/network/members
[stars-image]:https://img.shields.io/github/stars/maky-hnou/ML_OCR.svg
[stars-url]:https://github.com/maky-hnou/ML_OCR/stargazers
[license-image]:https://img.shields.io/github/license/maky-hnou/ML_OCR.svg
[license-url]:https://github.com/maky-hnou/ML_OCR/blob/master/LICENSE

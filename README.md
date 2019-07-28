# ML_OCR


## About this repo:  
This repository is a re-implementation of an OCR using Machine Learning.  
It has been developed on Ubuntu 18.04 using python 3.6 and TensorFlow 1.13.   
The original implementation is available via this [link](https://github.com/vinayakkailas/Deeplearning-OCR).


## Content:  

- **cnn_model:** a folder containing the needed files to build the RCNN.  
- **data:** a folder containing the char dicts.
- **data_provider:** a folder containing the needed file to prepare and provide the data to the RCNN.
- **global_configuration:** a folder containing the needed configuration for the RCNN.  
- **local_utils:** a folder containing the utils needed to manipulate the data, char dicts and to write the logs.  
- **tools:** a folder containing the needed files to generate the tfrecords, train the RCNN and to test it.  
- **generate_tfrecords.py:** the file to generate tfrecords.  
- **test_ocr.py:** the file to test the OCR.
- **train_ocr.py:** the file to train the RCNN.  
- **requirements.txt:** a text file containing the needed packages to run the project.


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

**3. Train the RCNN:**  
To train the shadownet, run `python train_ocr.py`.   
The trained model will be saved to a directory named `ML_OCR/model/shadownet/`.  

**4. Test the model:**  
To test the model, images test should be in tfrecord form (After generating tfrecords, you should have `test.tfrecord` file in `tfrec/`).
Add the model name to the path in `test_ocr.py`.    
You can test the model by running `python test_ocr.py`.  


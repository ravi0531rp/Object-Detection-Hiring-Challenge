# Object Detection Hiring Challenge

--------------------------------------------------------------------------------------------

This repository contains the files(codes + results) for the hiring challenge from a big
Tech Organisation.
The description is as follows:

--------------------------------------------------------------------------------------------
## Overview
* We have a **dataset** and we need to train an object detector to generate labels & 
  bounding boxes for 2 classes.
* The model chosen should have ***high precision(for boxes)*** & ***accuracy(  
  classification  of labels) & speed***
* The model chosen is **SSD Mobilenet V2 FPN**.

## Model Overview
* There are 2 parts in the model name. Let's go through each one of them.
* **SSD** -> This is a single-shot multibox detector algorithm which is used for object   
  localization and labelling.
  It is a significantly faster family of object detectors.
  It comprises of a base feature extractor network followed by additional conv layers using multiple feature maps.
  More details here. [Medium Blog](https://towardsdatascience.com/ssd-single-shot-detector-for-object-detection-using-multibox-1818603644ca)

* **Mobilenet** -> This is the feature extractor being used by SSD in this architecture. 
  We can use other architectures like VGG.
  It is a very lightweight image classification model used predominantly in Mobile Devices. 
  It uses depthwise separable convolutions to reduce the number of parameters.
  More details here. [Medium Blog](https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470)

## Assumptions
* The distribution is balanced. If there are very less number of bounding boxes for any class, then it will not produce a very accurate model.
* The bounding boxes are drawn with a consistent pattern, i.e., the padding for each box drawn is same when scaled.

## Framework
* I chose to use [Tensorflow Object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) as it provides the training process as a callable API with a [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) to choose the models from.	


## Preprocessing Stage

* There were two things as inputs 1) Annotations Json 2) Images folder
* I had to generate the dataset in the format required by the TF Object Detection API.
* Please refer to the [Image Splitter](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/Images%20Splitter.ipynb) ipynb file.
* Using that file, I shuffled and did the train-test split of the images.(80:20)

* After that, please refer to the [TF Records Generator](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/TF%20Records%20Generator.ipynb)
* Using that file, with some manipulation I generated the [train.csv](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/data/train.csv) and [test.csv](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/data/test.csv) as well as visualization.
* The labels are present in the [label_map](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/data/labelmap.pbtxt) file.
* After generating the csv files, I generated the .tfrecord files to be used by the TF Object detection API.

## Object Detection API Setup and Training

* Please refer to the [Colab Training Jupyter Notebook](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/Hiring_Challenge_EagleView_2.ipynb)
* First I cloned the repository on colab and set up all the dependencies.
* Then I downloaded the pretrained model and the config file from github.
* In the pipeline config file, I edited the 1) number of classes 2) extractor type 3) paths
  to the tfrecord files.
* Then, for around 13000 epochs with a very small ***batch size of 4***(due to limited compute), using the **model_main_tf2.py** script from the API 
  respository, I trained the SSD Model on our data.
* After that, I exported the model as a usable pb file to make further inferences.

## Inference Code
* Please refer to the [main.py](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/main.py) as the main inference script. 
* The config.py file has the paths to the saved model and label map to be used in the main file.
* The folder [utils](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/tree/master/utils) has two files [helpers.py](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/utils/Helpers.py) and [detector_num.py](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/utils/detector_num.py)
* Those two files are the helper files which are used in the main code for generating
  detections, drawing bounding boxes and stuff.


## Observations
* Since the free version of colab has limited threshold for inactivity as well as limited compute time, the model trained for < ***15k epochs***. The model did draw very nice bounding boxes, but IoU as well as the clasification precision can be improved.
* False positives can definitely be improved with further training.
* In the last epoch trained, this is the status obtained on the ***training set***.<br />
**{'Loss/classification_loss': 0.100076556,<br />
 'Loss/localization_loss': 0.05904458,<br />
 'Loss/regularization_loss': 0.12084896,<br />
 'Loss/total_loss': 0.2799701,<br />
 'learning_rate': 0.018827396}**

* Report obtained from evaluation on ***test*** dataset <br />
![Eval](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/outputImages/eval.png)

## Output on the Test Set from the [output Images folder](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/tree/master/outputImages)
* The format on the bounding boxes is (label, confidence)
* The threshold was set as 0.15 as the model was trained for a very small amount of time.
![Im1](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/outputImages/0.jpg)
![Im1](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/outputImages/1.jpg)
![Im1](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/outputImages/2.jpg)
![Im1](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/outputImages/3.jpg)
![Im1](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/outputImages/4.jpg)
![Im1](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/outputImages/5.jpg)
![Im1](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/outputImages/6.jpg)
![Im1](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/outputImages/7.jpg)
![Im1](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/outputImages/8.jpg)
![Im1](https://github.com/ravi0531rp/Object-Detection-Hiring-Challenge/blob/master/outputImages/9.jpg)
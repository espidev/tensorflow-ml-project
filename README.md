# tensorflow-ml-project

## Project Proposal

### Introduction

Recognizing various forms of land use via satellite imaging is an important means of measuring urban spread and human development. Using Tensorflow Keras, this project will use satellite images of urban and sub-urban settings to classify land use (e.g. freeway, agriculture, forest, etc.). 

The ultimate goal of this project is to take high-definition satellite images of whole cities and accurately generate statistics on land use.

### Data

The first part of our dataset comes from the UC Merced Land Use Dataset (http://weegee.vision.ucmerced.edu/datasets/landuse.html) which has 256x256 resolution images manually taken from the USGS National Map Urban Area Imagery collection. Images come from various sites across the United States. Each pixel represents one square foot. There are a total of 21 classes, with 100 images per class for a total of 2,100 images.

Citation: Yi Yang and Shawn Newsam, "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification," ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010.


The second part of our dataset comes from the NWPU-RESISC45 dataset (http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) which also has 256x256 resolution images created by the Northwestern Polytechnical University (NWPU). Images come from over 100 countries and a variety of regions. Within this dataset is 45 different classes, with 700 images per class for a total of 31,500 images. From a NWPU-RESISC45 dataset a few classes (e.g. palace, church, cloud, etc.) were omitted for being either irrelevant or too specific. 

Citation: G. Cheng, J. Han, X. Lu. Remote Sensing Image Scene Classification: Benchmark and State of the Art. Proceedings of the IEEE.


The third part of our dataset comes from senseFly (https://www.sensefly.com/education/datasets/?dataset=1502). Only a few hundred images were taken from the agricultural and airport datasets to bolster our image count.

In the end, there are a total of 26,920 images in our dataset with 34 classes as follows:

* agricultural    
* airplane        
* baseballdiamond 
* basketballcourt 
* beach
* bridge
* chaparral       
* denseresidential
* desert
* forest
* freeway
* golfcourse
* harbor
* industrial
* intersection
* island
* lake
* mediumresidential
* mobilehomepark
* mountain
* overpass
* parkinglot
* powerstation
* river
* roundabout
* runway
* seaice
* ship
* snowberg
* sparseresidential
* stadium
* storagetanks
* tenniscourt
* trackfield

### Data Processing
An important choice is whether or not colour should be factored into calculations. Since some land formations look very similar under greyscale, we will be including it. This will requiring normalization of all pixels into RGB, which means each pixel with have 3 values between 0 and 255.

### Training Method
We will be using multiple 2D convolutional layers in the model. We will start with larger kernel sizes in order to learn about larger features first, and then have later layers having smaller kernel sizes. All of these layers will have the relu activation function.

We can also add drop out layers between layers to reduce the effects of overtraining.

For the optimizer, we will likely use adam, because it works fairly well for other image classification problems.

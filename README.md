# tensorflow-ml-project

## Project Proposal

### Introduction

Recognizing various forms of land use via satellite imaging is an important means of measuring urban spread and human development. Using Tensorflow Keras, this project will use satellite images of urban and sub-urban settings to classify land use (e.g. freeway, agriculture, forest, etc.). 

The ultimate goal of this project is to take high-definition satellite images of whole cities and accurately generate statistics on land use.

### Data

Our data comes from the UC Merced Land Use Dataset (http://weegee.vision.ucmerced.edu/datasets/landuse.html) which has 256x256 resolution images manually taken from the USGS National Map Urban Area Imagery collection. Images come from various urban sites across the United States. Each pixel represents one square foot.

Citation: Yi Yang and Shawn Newsam, "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification," ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010.

The dataset contains 21 different classes of land use, with 100 images under each class for a total of 2,100 images. The different classes are as follows:

* agricultural
* airplane
* baseballdiamond
* beach
* buildings
* chaparral
* denseresidential
* forest
* freeway
* golfcourse
* harbor
* intersection
* mediumresidential
* mobilehomepark
* overpass
* parkinglot
* river
* runway
* sparseresidential
* storagetanks
* tenniscourt

### Data Processing
An important choice is whether or not colour should be factored into calculations. Since some land formations look very similar under greyscale, we will be including it. This will requiring normalization of all pixels into RGB, which means each pixel with have 3 values between 0 and 255.

### Training Method
We will be using multiple 2D convolutional layers in the model. We will start with larger kernel sizes in order to learn about larger features first, and then have later layers having smaller kernel sizes. All of these layers will have the relu activation function.

We can also add drop out layers between layers to reduce the effects of overtraining.

For the optimizer, we will likely use adam, because it works fairly well for other image classification problems.

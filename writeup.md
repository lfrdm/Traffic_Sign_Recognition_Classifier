# **Traffic Sign Recognition** 

---

**Classifying german traffic sign of the GTSRB with CNNs**
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with this written report

---

[//]: # (Image References)

[image0]: ./examples/ex1.png "Example images of training set"
[image1]: ./examples/hist.png "Histogram of classes"
[image2]: ./examples/data_aug.png "Data augmentation"
[image3]: ./examples/web_imgs.png "Images from Web"


## Reflection
In this report I summarize my work on this project and provide my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

## Data Set Summary & Exploration

### 1. Basic Summary of the Data Set

The numpy library was used to calculate summary statistics of the traffic
signs data set, which are the following:

| Data			        |     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Size training set      		| 34799 									| 
| Size validation set     			| 4410										|
| Size test set					| 4410									|
| Shape of traffic signs	      		| (32, 32, 3)				 				|
| Number of unique classes			| 43 							|

The given data set provides 32x32px RGB images as UINT8 of german traffic signs for classfification and was used in a competition calles "German Traffic Sign Recognition Benchmark" (GTSRB). 

### 2. Exploratory Visualization of Dataset

Three random example images of the training data and its corresponding labels are shown below.
![alt text][image0]

The histogram of classes shows the imbalance of class frequency.
![alt text][image1]

## Design and Test of Model Architecture

### 1. Preprocessing and Data Augmentation

For data preprocessing and augmentation it was decided to use nearly the same preprocessing LeCun & Sermanet used in Traffic Sign Recognition with Multi-Scale Convolutional Networks [source](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf):
* All images were converted from UINT8 to FLOAT32, because necessary for normalization.
* All images were normalized between 0 to 1 for each color channel separately with equation 6 of this [source](https://www.researchgate.net/publication/281118372_NumPy_SciPy_Recipes_for_Image_Processing_Intensity_Normalization_and_Histogram_Equalization).
* While training, the data was augmented by randomly scaling +/- 12.5%, rotating +/- 10° and translating +/- 1px in x- and y-direction. On one training example, four augmented were used.

The following image shows an original image of the training data set and 8 of its augmented versions:
![alt text][image2]

### 2. Model Architecture

The CNN architecture was structured as follows: 

| Layer         		|     Output Dimension     	        					| 
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							| 
| Convolution   	    | 32x32x32 	|
| Convolution				|	32x32x32											|
| Max pooling	      	| 16x16x32 			|
| Convolution 	    | 16x16x64      									|
| Convolution		| 16x16x64        									|
| Max pooling				| 8x8x64        									|
| Convolution					|	8x8x128											|
| Convolution						|		8x8x128											|
 | Max pooling 	    |       				4x4x128	(2048 flattened)					|
| Fully connected		| 1024        									|
| Fully connected| 1024       									|
| Output					|	43 Classes										|

* All convolutional layers had a kernel of 3x3xc and included a ReLU as activation function. 
* The max pooling operations had a 2x2 kernel and a stride of 2. 
* The fully connected layers were followed by a ReLU activation and a Dropout operation afterwards with 50% keep probability.

### 3. Training Procedure

The training procedure of the given LeNet was adapted:
* **Optimizer:** The Adam optimizer was used with its standard learning rate of alpha = 0.001, beta1 = 0.9 and beta2 = 0.999 [source](https://arxiv.org/pdf/1412.6980.pdf). Adam optimizer was used, because it updates the learning rate and uses momentum when updating the weights.
* **Epochs:** 30 epochs were chosen ruffly based on the increase of accuracy on the validation set, around 15 epochs would have been enough due to only minor increases in accuracy.
* **Batch size:** A batch size of 128 was chosen to be able to train on my local machine with CPU.

### 4. Approach

Results on data sets:
* Training set accuracy was not calculated, due to the fact, that a model which is big enough will always fit to the training data with nearly 100% accuracy.
* Validation set accuracy of **99.3%** was reached.
* Test set accuracy of **97.6%** with the last model saved was reached, with a model earlier in training, an test accuracy of **98.1%** was accomplished.

Iterative approach:
* It was started with the LeNet architecture, but it was too small to reach a high enough accuracy.
* With normalization and data augmentation the LeNet architecture reached accuracies around 92% of the validation set.
* The number of filters were increased to create better features (32, 64, 128) and an extra 3rd conv layer with ReLU (after an extra max pooling operation) was added to increase representation of feature maps. Two fully connected layers used standard amount of 1024 neurons. Accuracy on validation increased to around 95%.
* Added extra conv layer per conv block (before subsampling) and included Dropout with 50% keep probability in fully connected layers. This gave an big increase in performance due to regularization of the classifier.
* This type of architecture is a mix between the standard CNN architecture of the LeNet and the VGG-13 network [source](https://arxiv.org/pdf/1409.1556.pdf), except that they use one more conv block.
* The accuracy shows, that chosen type of network, data augmentation and preprocessing works very well and reaches state of the art accuracy on the test set.

## Test a Model on New Images

### 1. Testing on Unknown Images

Five images were randomly selected from the web, which are shown below:
![alt text][image3]

All images were classified correctly, which is very surprising, due to the fact, that they are different from the data, the network was trained with. The network already was trained with some translation-, rotation- and scaling-invariance due to data augmentation and max pooling, but only to a certain degree.

### 2. Model Prediction 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (70km/h)  									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| Road work					| Road work											|
| Stop	      		| Stop					 				|
| Yield			| Yield|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%, which is comparable to the results of the test set.

### 3. Model Prediction with Softmax

The code for making predictions on my final model is located in the 9th cell of the Ipython notebook.

For the first image, the model is relatively unsure that this is a Speed limit (70km/h) sign (probability of 0.38). It is very tilted, but not in the image-axis, which is probably the reason for the low softmax prediction, the data in the training set were all frontal images.
For the second image, the model is 100% sure and correct.
For the third image, the model is 100% sure and correct.
For the fourth image, the model is relatively unsure that this is a Stop sign (probability of 0.56). The stop sign objective is small relative to the image patch, due to the white border.
For the fifth image, the model is near 100% sure and correct.

Image | Probability         	|     Prediction	        					| 
------|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)| .3881         			| Speed limit (70km/h)   									| 
| | .3830     				| Speed limit (60km/h)										|
| | .1213					| Double curve										|
| | .0309	      			| General caution					 				|
| | .0307				    | Traffic signals     							|
| Right-of-way at the next intersection| 1.0000         			| Right-of-way at the next intersection  									| 
| | .0000     				| Pedestrians										|
| |.0000					| Double curve											|
| |.0000	      			| Ahead only					 				|
| |.0000				    | Road work      							|
| Road work| 1.0000         			| Road work  									| 
| | .0000     				| Road narrows on the right										|
| |.0000					| Dangerous curve to the right											|
| |.0000	      			| Bumpy Road					 				|
| |.0000				    | Bicycles crossing      							|
| Stop| .05582         			| Stop									| 
| | .4386     				| Priority road										|
| |.0022					| No entry											|
| |.0004	      			| Roundabout mandatory					 				|
| |.0001				    | Right-of-way at the next intersection      							|
| Yield| .9998         			| Yield									| 
| | .0002     				| Priority road										|
| |.0000					| No vehicles											|
| |.0000	      			| No passing					 				|
| |.0000				    | Keep right      							|

## (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



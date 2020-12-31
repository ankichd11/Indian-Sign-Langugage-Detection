# Indian-Sign-Langugage-Detection
Sign Language Detection project is based on the real life problems for deaf and dumb people who use sign language to communicate. There are very few people who can understand sign language and thus makes it difficult for the deaf and dumb people to communicate. In this project, we are going to recognize sign language using hand gestures which will make it easier for the handicapped people to communicate with people who do not understand sign language. The aim is to build a human computer interface which can solve the above described problem in the simplest way possible, and with great accuracy. 

### <p align="center">METHODOLOGY</p>

Project is based on the concept of computer vision and it also eliminates the problem of using any artificial device for interaction as all the signs or gestures are represented with bare hands. 

#### DatasetGeneration
As less research has been done for the Indian Sign Language as compared to ASL proper dataset is not available for ISL, so we have prepared our own dataset. We have built a python file through which we can generate our data for all the Classes. So  for creating a dataset we have to use the Open Computer Vision(OpenCV) library. Firstly we captured around 7000 total images 200 for each 35 labels ISL. Then we divided the dataset in 80:20 percent ratio into training and testing data respectively.  

#### Gesture Classification

Our Approach for Sign language classification:
Our approach used following steps for final detection of symbol

Algorithm Step 1: Image Segmentation

Algorithm Step 2: Feature extraction 

Algorithm Step 3: SVM Model for Classification 

#### <p align="center">Step 1: Image Preprocessing</p>

#### Image Segmentation

The goal of Image segmentation is to remove background and noises or we can say simplify and/or change the representation of an image into something which is Region of Interest (ROI) and the only useful information in the image. Image segmentation is typically used to locate objects and boundaries (lines, curves, etc.) in images.

Image segmentation can be achieved when following steps are performed:

<b>Skin Masking:</b> Using the concept of thresholding this RGB color space is converted into grayscale image and SkinMask is finally obtained through HSV color space(which we get from gray scale image)

<b>Canny edge detection:</b> It is basically a technique which identifies or detects the presence of sharp discontinuities in an image there by detecting the edges of the figure in focus.

It is multi step algorithm which is followed as:

Step 1: Computing the horizontal (Gx) and vertical (Gy) gradient of each pixel in an image.

Step 2: Using the above information the magnitude (G) and direction (of each pixel in the image is calculated.

Step 3: In this step all non-maxima‟s are made as zero that is suppressing the non- maxima‟s thus the step is called Non-Maximal Suppression.

Step 4: The high and low thresholds are measured using the histogram of the gradient magnitude of the image

Step 5: To get the proper edge map hysteresis thresholding is employed which will link between the weak and strong edges. The weak edges are taken into consideration if and only if it is connected to one of the strong edges or else it is eliminated from the edge map. The strong edge is the one whose pixel is greater than the high threshold and weak edge is one whose pixel value lies between high and low threshold. 

![alt text](https://github.com/ankichd11/Indian-Sign-Langugage-Detection/blob/main/image1.jpg?raw=true)                      
<p align="center">Fig 1:  Different stages of Image Preprocessing.</p>


#### <p align="center">Step2: Feature Extraction</p>

In feature extraction following steps were performed:

<b>Feature detection :</b> key features of the image were extracted using SURF technique. SURF is a feature extraction algorithm which is robust against rotation variation scaling.
We have extracted features using the inbuilt SURF function in opencv.

<b>Clustering:</b> To cluster all the features obtained in the above step we apply mini batch k- means clustering(similar to K-means clustering but efficient in terms of time consumption and memory).

In our code we have taken k as 8 for the image of each label. So the cluster size or total number of feature descriptors are 8 * 35. After training of all SURF features (extracted in above step) through mini batch k-means clustering, all similar features are clustered in a cluster. Total number of clusters are also known as visual words.
So in this step we obtained visual words for each image.

Histogram Computation: In this step we computed Histogram using predicted visual words(generated above). This is done by calculating the frequency of each visual word belonging to the image in total visual words.

#### <p align="center">Step 3: Classification</p> 
#### SVM Model for Classification:

Once all the  histograms are generated for the total data set using the above step, the training dataset is trained using Support Vector Machine Classifier and then predicted with a linear kernel. Other Classifiers like CNN, KNN, Logistic Regression can also be used for classification.

Additionally for the purpose of real time recognition the trained model is saved in a file so that a user can predict the gesture using video feed in real time.

### <p align="center">TRAINING AND TESTING</p>

First we convert our data from RGB to grayscale then after applying medianblur filter(to get skin masked image) canny edge detection algorithm is applied to all images using inbuilt Canny method. After this SURF algorithm is applied to all images to get feature descriptors. 

Once feature descriptors are obtained for all images these are trained over the clustering model, and histograms are computed, these training data histograms mapped with their respective labels and  are trained over SVM classifier and once training is done histograms generated labels for testing dataset are predicted using SVM classifier predict method.

### <p align="center">RESULTS</p>

We scored a 99% accuracy using the Bow, integrated with robust SURF feature descriptors. The real time recognition prediction of results can be seen by the figure below. Using a large set of data images always helps us to get a better efficiency in result as there could be slight biasing in the model prediction as the data set has much similar images without variations.

We also created a file to generate the confusion matrix for our model which is shown in below figure 

![alt text](https://github.com/ankichd11/Indian-Sign-Langugage-Detection/blob/main/image2.jpg?raw=true)
<p align="center">Fig 2: Confusion matrix for our model.</p>			

We can see that almost 99 % of labels are predicted correctly. Some labels with incorrect prediction are G,T,3,4 which are wrongly predicted with S,U ,2,3respectively.

The 40 in the confusion matrix represents the total number of testing dataset for each label which is 20%  of 200.

Metrics showing Precision score, Recall score and F1 score for each label are attached as follows

![alt text](https://github.com/ankichd11/Indian-Sign-Langugage-Detection/blob/main/image3.jpg?raw=true)
<p align="center">Fig 3: Accuracy metrics</p>


#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image/1.png "Visualization"
[image2]: ./image/2.png "training distribution"
[image3]: ./image/3.png "german traffic signs"
[image4]: ./image/4.png "Traffic Sign prediction"
[image5]: ./image/5.png "Traffic Sign prediction distribution"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yang1688899/CarND-Traffic-Sign-Classifier-Project.git/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set:
![alt text][image1]

It is a bar chart showing distribution of the training data:

![alt text][image2]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to a single channel img, it helps to reduce training time and filter the unrelated information.
```
#Convert to single channel Y
data = 0.299 * data[:, :, :, 0] + 0.587 * data[:, :, :, 1] + 0.114 * data[:, :, :, 2]
```

Second, i scale the values of img to [0,1]
```
#Scale features to be in [0, 1]
data = (data / 255.).astype(np.float32)
```

As a last step, I sharpen the image data using skiamge lab
```
#sharpen image
for i in range(data.shape[0]):
    data[i] = exposure.equalize_adapthist(data[i])
```


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 image   							    | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU          		|       									    |
|Max pooling     		| 2x2 stride,  outputs 5x5x16					|
|Flatten				| outputs 400									|
|Fully Connected    	| outputs 120					                |
| RELU          		|       									    |
|Fully Connected    	| outputs 84					                |
| RELU          		|       									    |
|Fully Connected    	| outputs 43					                |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with learning rate 0.001, batch size 128 and 50 epochs

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

At first I choose LeNet architecture. This architecture was intruduce to me to classify handwriting digit in lession 8, which it prove to be a very powerful architecture for dealing with image recognition.

The LeNet architecture take 32x32x3 image as input, which isn't match what the data I used after precessing, which is 32x32x1 image, so I have to change the architeture to make it accept 32x32x1 image as input. And I also change the output to be 43 classes instead of 10.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.943
* test set accuracy of 0.967
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are German traffic signs that I found on the web:

![alt text][image3]

All the image are taken at night and some of them have flash point on it, which will make it a lot more harder for the model to recognize.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image4]


The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. The first one misclassified Speed Limit(20km/h) to Speed Limit(30km/h), the second one misclassified Speed Limit(60km/h) to Speed Limit(50km/h). Seem the model is a little bit confused about the diffrence of diffrent Speed Limit traffic signs.

Compare with the test set accuracy of 0.967, the accuracy on the new image seems to be too low, but consider it's only 10 sample, it is hard to say that the model is not doing well.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model：
```
predict = tf.argmax(logits,1)
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('lenet.meta')
    saver.restore(sess, './lenet')
    prediction = sess.run(predict,feed_dict={x:new_data_processed})
```

The accuracy 
The softmax probabilities of the new images:

![alt text][image5]

For the first image, the model is defintely sure that this is a go straight or left sign (above 0.9 probability), and the image does contain a go straight or left sign.

For the second image, the model is defintely sure that this is a keep right sign (probability of 1.0), and the image does contain a keep right sign.

For the third image, the model is defintely sure that this is a no entry sign (probability of 1.0), and the image does contain a no entry sign.

For the fourth image, the model is defintely sure that this is a no vehicle sign (close to probability of 1.0), and the image does contain a no vehicle sign.

For the fifth image, the model is defintely sure that this is a priority road sign (probability of 1.0), and the image does contain a priority road sign.

For the sixth image, the model is defintely sure that this is a yield sign (probability of 1.0), and the image does contain a yield sign.

For the seventh image, the model is confused. It predict 0.53 probability for  speed limit(30km/h) and 0.45 for speed limit(80km/h), while the image actually contain a speed limit(20km/h). Thing become really wired while the model dealing with speed limit sighs.

For the eighth image, the model predict 0.82 probability for ahead only and 0.18 for go straight or right, while the image actually contain a ahead only sign. The model make a petty good prediction in this case.

For the nineth image, the model pridict 1.0 probability for speed limit(60km/h) while the image actually contain a speed limit(50km/h) sign. The model is totally wrong is this case.

For the tenth image, the model pridict 1.0 probability for speed limit(80km/h) while the image actually contain a speed limit(80km/h) sign. At last a speed limit sigh is rightfully classfied by the model.

From what we have above, the model seems to be doing a great job to classified the sign image except the speed limit signs.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



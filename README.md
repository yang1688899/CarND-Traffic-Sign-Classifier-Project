## **交通标志识别**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


**实现步骤：**

* 加载数据
* 分析，预处理数据
* 设计，训练，测试模型
* 使用模型测试对新的图片进行预测

[//]: # (Image References)

[image1]: ./image/1.png "Visualization"
[image2]: ./image/2.png "training distribution"
[image3]: ./image/3.png "german traffic signs"
[image4]: ./image/4.png "Traffic Sign prediction"
[image5]: ./image/5.png "Traffic Sign prediction distribution"

### 加载数据

由于提供的数据存放在pickle文件中，直接使用pickle加载即可:

```
training_file = 'I:/traffic-signs-data/train.p'
validation_file= 'I:/traffic-signs-data/valid.p'
testing_file = 'I:/traffic-signs-data/test.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

### 分析，预处理数据
加载的数据包含了34799张训练图片(training set)，4410张校验图片(validation set)，12630张测试图片(test set)，均为32x32x3的图片，共43个分类。

以下为各分类下的图片展示:
![alt text][image1]

以下为训练数据集的的直方分布图:

![alt text][image2]


训练图片有rgb三个颜色通道，这里把它转换为单颜色通道图片，这样可以减少训练时间已经提升模型的泛用性：
```
#Convert to single channel Y
data = 0.299 * data[:, :, :, 0] + 0.587 * data[:, :, :, 1] + 0.114 * data[:, :, :, 2]
```

图片的每个像素值的区间为[0,255],这里需要把它normalize为值区间在[0,1]，以便更好的训练模型。
```
#Scale features to be in [0, 1]
data = (data / 255.).astype(np.float32)
```

其中的一些图片的交通标志轮廓模糊，这里使用skimage库的equalize_adapthist()来增强图片对比度，从而使交通标志轮廓更明晰。
```
#sharpen image
for i in range(data.shape[0]):
    data[i] = exposure.equalize_adapthist(data[i])
```


### 设计，训练，测试模型

最终的模型结构如下:

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

以下为实现代码:
```
#initialize weights
def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

#initialize bias
def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)

#conv2b
def conv(x, w, b):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'VALID')+b

#2x2 maxpooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
```
```
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
# Hyperparameters
mu = 0
sigma = 0.1

# Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
conv1_w = weight_variable(shape=(5, 5, 1, 6))
conv1_b = bias_variable(shape=[6])
conv1 = conv(x,conv1_w,conv1_b)

#Activation
conv1_act = tf.nn.relu(conv1)

# Pooling. Input = 28x28x6. Output = 14x14x6.
pool1 = max_pool_2x2(conv1_act)

# Convolutional. Output = 10x10x16.
conv2_w = weight_variable(shape=(5, 5, 6, 16))
conv2_b = bias_variable(shape=[16])
conv2 = conv(pool1,conv2_w,conv2_b)

#Activation
conv2_act = tf.nn.relu(conv2)

# Pooling. Input = 10x10x16. Output = 5x5x16.
pool2 = max_pool_2x2(conv2_act)

# Flatten. Input = 5x5x16. Output = 400.
f1 = flatten(pool2)

# Layer 3: Fully Connected. Input = 400. Output = 120.
fc1_w  = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
fc1_b = tf.Variable(tf.zeros(120))
fc1 = tf.nn.relu( tf.matmul(f1, fc1_w) + fc1_b )

# Layer 4: Fully Connected. Input = 120. Output = 84.
fc2_w  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
fc2_b = tf.Variable(tf.zeros(84))
fc2 = tf.nn.relu( tf.matmul(fc1, fc2_w) + fc2_b )

# Fully Connected. Input = 84. Output = 43.
fc3_w  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
fc3_b = tf.Variable(tf.zeros(43))
logits = tf.matmul(fc2, fc3_w) + fc3_b
```
 


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



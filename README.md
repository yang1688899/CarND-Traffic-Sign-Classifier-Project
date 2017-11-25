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
[image6]: ./image/noEntryO.png
[image7]: ./image/notEntry.png
[image8]: ./image/speedLimmit20.png
[image9]: ./image/speedLimmit20O.png
[image10]: ./image/y_channel.png

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


训练图片为rgb三个颜色通道，这里把它转换为YUA颜色空间（color space）的Y通道图片，这样可以减少训练时间以及提升模型的泛用性：
```
#Convert to single channel Y
data = 0.299 * data[:, :, :, 0] + 0.587 * data[:, :, :, 1] + 0.114 * data[:, :, :, 2]
```
以下为转换前后对比:

![alt text][image10]

图片的每个像素值的区间为[0,255],这里需要把它normalize为值区间在[0,1]，以便更好的训练模型。
```
#Scale features to be in [0, 1]
data = (data / 255.).astype(np.float32)
```

### 设计，训练，测试模型

最终的模型结构如下:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 image   							    | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x108 	|
| Tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x108 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x200   |
| Tanh          		|       									    |
|Max pooling     		| 2x2 stride,  outputs 5x5x200					|
|Flatten				| outputs 47052									|
|Fully Connected    	| outputs 50					                |
| Tanh          		|       									    |
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

# Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x108.
conv1_w = weight_variable(shape=(5, 5, 1, 108))
conv1_b = bias_variable(shape=[108])
conv1 = conv(x,conv1_w,conv1_b)

#Activation
conv1_act = tf.nn.tanh(conv1)

#Pooling. Input = 28x28x108. Output = 14x14x108.
pool1 = max_pool_4x4(conv1_act)

# Convolutional. Output = 10x10x200.
conv2_w = weight_variable(shape=(5, 5, 108, 200))
conv2_b = bias_variable(shape=[200])
conv2 = conv(conv1,conv2_w,conv2_b)

#Activation
conv2_act = tf.nn.tanh(conv2)

#Pooling. Input = 10x10x200. Output = 5x5x200.
pool2 = max_pool_2x2(conv2_act)

# Flatten. Input = 5x5x200. Output = 47052.
f1 = flatten(pool1)
f2 = flatten(pool2)
f_conct = tf.concat([f1,f2],1)

# Layer 3: Fully Connected. Input = 47052. Output = 50.
fc1_w  = tf.Variable(tf.truncated_normal(shape=(47052, 50), mean = mu, stddev = sigma))
fc1_b = tf.Variable(tf.zeros(50))
fc1 = tf.nn.tanh( tf.matmul(f_conct, fc1_w) + fc1_b )

# Fully Connected. Input = 84. Output = 43.
fc2_w  = tf.Variable(tf.truncated_normal(shape=(50, 43), mean = mu, stddev = sigma))
fc2_b = tf.Variable(tf.zeros(43))
logits = tf.matmul(fc1, fc2_w) + fc2_b
```

使用学习率(learning rate)为0.001的AdamOptimizer来训练模型，损失函数(loss function)为cross entropy, batch size为128, epochs为20。
```
rate = 0.001

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    session = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = session.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

EPOCHS = 50
BATCH_SIZE = 128

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            session.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(session, './lenet')
    print("Model saved")
```

最终在测试数据集得到94.1%的准确率。


### 使用模型测试对新的图片进行预测
以下为网上找的10张交通标志图片：

![alt text][image3]

图片的光照环境较差，且有一些图片有明显的闪光点。

以下为图片的预测结果：

![alt text][image4]

以下为各图片的前5个预测：

![alt text][image5]


十张图片准确预测了七张，其中错误的预测均为速度限制类交通标志,可以看出模型在辨识交通标志中的数字方面表现并不理想

以下为第一层CNN捕捉到的"NO ENTRY"交通标志的特征图:

原图：
![alt text][image6]

特征图：
![alt text][image7]

以下为第一层CNN捕捉到的"SPEED LIMMITS 20KM/H"交通标志的特征图：

原图：
![alt text][image8]

特征图：
![alt text][image9]

可以看到模型对简单的几何图形特征可以清晰捕捉，但对于数字则有点模糊了




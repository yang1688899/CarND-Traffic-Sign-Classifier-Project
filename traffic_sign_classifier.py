# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 14:34:38 2017

@author: yang
"""

import pickle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import random
from skimage import exposure
import os
import matplotlib.image as mpimg

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'I:/traffic-signs-data/train.p'
validation_file= 'I:/traffic-signs-data/test.p'
testing_file = 'I:/traffic-signs-data/valid.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples

n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

# Visualizations will be shown in the notebook.

with open('signnames.csv') as csvfile:
    class_name = []
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_name.append(row['SignName'])
class_nums = [] 
plt.figure(figsize=(14,150))
for i in range(n_classes):
    
    X_class = X_train[y_train==i]
    show_sample_index = random.sample(range(len(X_class)), 5)
    for j in range(5):
        plt.subplot(43,5,5*i+j+1)
        plt.imshow(X_class[show_sample_index[j]])
    title = '%s: %s'%(i,class_name[i])
    plt.title(title)
    
    class_nums.append(len(X_class))

#distribution of classes in the training

plt.figure()
plt.bar(range(n_classes), class_nums)
plt.xlabel("Class")
plt.ylabel("Sample Number")
plt.show()

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

def preprocess(data):

    #Convert to single channel Y
    data = 0.299 * data[:, :, :, 0] + 0.587 * data[:, :, :, 1] + 0.114 * data[:, :, :, 2]
    #Scale features to be in [0, 1]
    data = (data / 255.).astype(np.float32)
        
    for i in range(data.shape[0]):
        data[i] = exposure.equalize_adapthist(data[i])
    
        
    data = data.reshape(data.shape + (1,))
    
    return data

X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_test = preprocess(X_test)

print('preprocess is done!')


### Define your architecture here.
### Feel free to use as many code cells as needed.
def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    layer1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    layer1_b = tf.Variable(tf.zeros(6))
    layer1 = tf.nn.conv2d(x,layer1_w,strides=[1,1,1,1],padding='VALID')+layer1_b
    # Activation.
    layer1 = tf.nn.relu(layer1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolutional. Output = 10x10x16.
    layer2_w = tf.Variable(tf.truncated_normal(shape=(5,5,6,16), mean=mu, stddev=sigma ))
    layer2_b = tf.Variable(tf.zeros(16))
    layer2 = tf.nn.conv2d(layer1 , layer2_w, strides=[1,1,1,1], padding='VALID')+layer2_b
    
    # Activation.
    layer2 = tf.nn.relu(layer2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    layer2 = tf.nn.max_pool(layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    # Flatten. Input = 5x5x16. Output = 400.
    layer2 = flatten(layer2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    layer3_w = tf.Variable(tf.truncated_normal(shape=(400,120), mean = mu, stddev = sigma))
    layer3_b = tf.Variable(tf.zeros(120))
    layer3 = tf.matmul(layer2,layer3_w)+layer3_b

    # Activation.
    layer3 = tf.nn.relu(layer3)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    layer4_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    layer4_b = tf.Variable(tf.zeros(84))
    layer4 = tf.matmul(layer3, layer4_w) + layer4_b

    # Activation.
    layer4 = tf.nn.relu(layer4)
    
    # Fully Connected. Input = 84. Output = 43.
    layer5_w  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    layer5_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(layer4, layer5_w) + layer5_b

    return logits


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

logits = LeNet(x)
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
    
    
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('lenet.meta')
    saver.restore(sess, './lenet')
    
    training_accuracy = evaluate(X_train, y_train)
    print("Training Accuracy = {:.3f}".format(training_accuracy))
    
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
    
import os
import matplotlib.image as mpimg

img_names = os.listdir("new-traffic-signs/")
new_data = []
plt.figure()
i = 1
for name in img_names:
    path = "new-traffic-signs/"+name
    img = mpimg.imread(path)
    
    plt.subplot(2,5,i)
    plt.imshow(img)
    i+=1
    
    new_data.append(img)
    
new_data = np.asarray(new_data)
new_data_processed = preprocess(new_data)
new_labels = [37, 38, 17, 15, 12, 13, 0, 35, 3, 5]


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
predict = tf.argmax(logits,1)
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('lenet.meta')
    saver.restore(sess, './lenet')
    prediction = sess.run(predict,feed_dict={x:new_data_processed})

print (prediction)

plt.figure(figsize=(12,16))
for i in range(len(prediction)):
    plt.subplot(4,3,i+1)
    title = 'pridect:Class-%s %s'%(prediction[i],class_name[prediction[i]])
    plt.title(title)
    plt.imshow(new_data[i])
    
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('lenet.meta')
    saver.restore(sess, './lenet')
    test_accuracy_on_new_img = evaluate(new_data_processed, new_labels)
    print("Test on new image accuracy = {:.3f}".format(test_accuracy_on_new_img))
    
    
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
top_5 = tf.nn.top_k(tf.nn.softmax(logits),5)
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('lenet.meta')
    saver.restore(sess, './lenet')
    top_5_predicts = sess.run(top_5,feed_dict={x:new_data_processed})
predict_indices = top_5_predicts.indices
predict_values = top_5_predicts.values

plt.figure(figsize=(12,20))
for i in range(len(new_data)):
    
    plt.subplot(10,2,2*i+1)
    plt.imshow(new_data[i])
    
    indices = predict_indices[i]
    values = predict_values[i]
    predict_class = [class_name[i] for i in indices]
    plt.subplot(10,2,2*i+2)
    
    
    y_pos = np.arange(len(predict_class))

    plt.barh(y_pos, values, align='center', alpha=0.4)
    plt.yticks(y_pos, predict_class)
        
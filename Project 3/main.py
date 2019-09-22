#----------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------CNN----------------------------------------------------------------#

import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline  
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 100
print(y_train[image_index]) # The label is 5
plt.imshow(x_train[image_index], cmap='Greys')

x_train.shape

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape)) #batch size=28
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(130, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

history=model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
history=model.fit(x=x_train,y=y_train, epochs=10)

model.evaluate(x_test, y_test)
model.predict(x_test)

#plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['acc'])
#plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------Random Forest----------------------------------------------------------------#

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from pylab import *

mnist = fetch_mldata('MNIST original')
n_train = 60000
n_test = 10000
indices = arange(len(mnist.data))
train_idx = arange(0,n_train)
test_idx = arange(n_train+1,n_train+n_test)
X_train, Y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, Y_test = mnist.data[test_idx], mnist.target[test_idx]

#RandomForestClassifier
classifier2 = RandomForestClassifier(10); 
classifier2.fit(X_train, Y_train) 

x=classifier2.predict(X_test)
y=classifier2.predict(X_train)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test,x)
print(confusion_matrix)

%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
plt.plot(y,'o', color='black');

#----------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------SVM----------------------------------------------------------------#

# SVM & RandomForest
import numpy as np
from pylab import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata

mnist1 = fetch_mldata('MNIST original')
n_train1 = 60000
n_test1 = 10000
indices1 = arange(len(mnist1.data))
train_idx1 = arange(0,n_train1)
test_idx1 = arange(n_train1+1,n_train1+n_test1)
X_train1, y_train1 = mnist1.data[train_idx1], mnist1.target[train_idx1]
X_test1, y_tes1t = mnist1.data[test_idx1], mnist1.target[test_idx1]

# SVM
classifier1 = SVC(kernel='rbf', C=2, gamma = 0.001)
classifier1.fit(X_train1, y_train1)

#----------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------Logistic Regression-------------------------------------------------------------#

from sklearn.datasets import fetch_mldata
#mnist = fetch_mldata('MNIST original')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist2     = input_data.read_data_sets('data/', one_hot=True)
trainimg   = mnist2.train.images
trainlabel = mnist2.train.labels
testimg    = mnist2.test.images
testlabel  = mnist2.test.labels
print ("MNIST loaded")

x = tf.placeholder("float", [None, 784]) 
y = tf.placeholder("float", [None, 10])  # None is for infinite 
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# LOGISTIC REGRESSION MODEL
actv = tf.nn.softmax(tf.matmul(x, W) + b) 
# COST FUNCTION
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1)) 
# OPTIMIZER
learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# PREDICTION
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))    
# ACCURACY
accr = tf.reduce_mean(tf.cast(pred, "float"))
# INITIALIZER
init = tf.initialize_all_variables()

import sys
training_epochs = 10
batch_size      = 100
display_step    = 5

# SESSION
sess = tf.Session()
sess.run(init)
# MINI-BATCH LEARNING
for epoch in range(training_epochs):
    avg_cost = 0.
    num_batch = int(mnist2.train.num_examples/batch_size)
    for i in range(num_batch): 
        batch_xs, batch_ys = mnist2.train.next_batch(batch_size)
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        feeds = {x: batch_xs, y: batch_ys}
        avg_cost += sess.run(cost, feed_dict=feeds)/num_batch
    # DISPLAY
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y: batch_ys}
        feeds_test = {x: mnist2.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print ("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f" 
               % (epoch, training_epochs, avg_cost, train_acc, test_acc))
print ("DONE")


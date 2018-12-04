from __future__ import print_function
import tensorflow as tensorflow
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import *
from keras import backend as K
from keras.models import load_model
from keras import optimizers
import numpy as np
from skimage import io
from keras.preprocessing import image
import h5py
import cv2

img_rows = 28
img_cols = 28
batch_size = 128
epochs = 3
learning_rate = 0.001
nr_classes = 10

#Loading and formating data
mnist = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


#Preprocessing images i.e normalize the data to take values from [0-1] and not from [0-255] for simplicity
x_train = tensorflow.keras.utils.normalize(x_train, axis=1)
x_test = tensorflow.keras.utils.normalize(x_test, axis=1)
#print(x_train[0])


#Building the architecture
model = tensorflow.keras.models.Sequential()
#Convolution layer with 32 output filters, a kernel size of 3x3
model.add(tensorflow.keras.layers.Conv2D(32,kernel_size=(3,3), input_shape=input_shape, data_format = 'channels_last'))
#Convolution layer with 64 output filters, a kernel size of 3x3
model.add(tensorflow.keras.layers.Conv2D(64,kernel_size=(3,3)))
#Maxpool layer with pool size 2x2
model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#Dropout Layer with rate = 0.5 (dropout fraction)
model.add(tensorflow.keras.layers.Dropout(0.5))
#Flatten Layer
model.add(tensorflow.keras.layers.Flatten())
#Fully connected layer with 128 neurons (outputs)
model.add(tensorflow.keras.layers.Dense(128))
#Dropout Layer with rate = 0.5 (dropout fraction)
model.add(tensorflow.keras.layers.Dropout(0.5))
#Fully-connected layer with as many neurons as there are classes in the problem (Output layer) i.e 10, activation function: Softmax
model.add(tensorflow.keras.layers.Dense(nr_classes, activation='softmax'))

#Setting up the model 
optimizer = tensorflow.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

loss_value, accuracy_value = model.evaluate(x_test, y_test)
print("Loss value: ", loss_value, "Accuracy value: " , accuracy_value)


#Probability distributions
predictions = model.predict([x_test])
#print(predictions)

#Predicting the given image
print("Processing image")   
img = image.load_img(path="digit.png",color_mode="grayscale",target_size=(28,28,1))
#plt.imshow(img, cmap=plt.cm.binary)
#plt.show()
img = tensorflow.keras.utils.normalize(img, axis=1)
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img_class = model.predict_classes(img)
prediction = img_class[0]
print("Class: ", prediction)


#To save the model in as .h5 file
#model.save('digit_recognition_model.h5')
#my_model = tensorflow.keras.models.load_model('digit_recognition_model.h5')
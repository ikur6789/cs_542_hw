#!/usr/bin/env python
# coding: utf-8

# ## Ian Kurzrock Assignment 8

# Importing neccesary modules
# Need keras framework, import keras
# fashion_mnist is our data
# sequential allows for the creation of a nerual network
# dense, droput, and flatten are all layers to be added to the network
# conv2D, and maxpooling2D are layers as well
# 

# In[1]:


import keras

from keras.datasets import fashion_mnist

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# #### Hyper parameters

# Hyper parameters are not learned by the model and are given by us.
# Batch_size is how many to be taught at time. Num_classes, number of classifications for items. Epochs, iteration times the model will learn on the data. img_rows and img_cols is the size of the images, necessary for importing.

# In[2]:


batch_size = 128

num_classes = 10

epochs = 12

img_rows, img_cols = 28, 28


# #### Loading in Dataset

# Python script to import data

# In[3]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# #### Checking shape of data downloaded

# Also creating ..reset data types to return tested data back for next optimizer

# In[4]:


x_train.shape
x_train_reset = x_train


# In[5]:


y_train.shape
y_train_reset = y_train


# In[6]:


x_test.shape
x_test_reset = x_test


# In[7]:


y_test.shape
y_test_reset = y_test


# #### Function to display

# Will now show some images in the dataset. cols_show is how many pictures to be shown.

# In[8]:


def display_image(data, rows, cols, index):
    plt.subplot(rows, cols, index)
    plt.imshow(data, cmap='gray')


# #### Display 10 images from the dataset

# In[9]:


cols_show = 10
plt.figure(figsize=(28,28)) #Images are 28 by 28

for i in range(cols_show):
    display_image(x_train[i], 1, cols_show, i+1)


# Determining if channels are needed first or last and creating a model input shape

# In[10]:


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# In[11]:


input_shape


# #### Normalize Pixel Values to be [0.0,1.0]

# Assinging floats so division can be performed. Dividing by 255 to get between 0.0-1.0

# In[12]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


# Sanity check, making sure values are being calculated correctly. (Contents are hidden for organization)

# In[13]:


for i in range(0,28):
    for j in range(0,28):
        for k in range(0,28):
            if x_train[i][j][k] > 0.0:
                print(x_train[i][j][k])
            


# Keras function to one hot encode data to num of classes

# In[14]:


#One hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)


# ### Build Deep Learning Model

# Refer to code comments. Sequential starts, add the 3 conv layers, max pool, and then 2 dense layers. Where dropouts are flattens are necessary for the model.

# In[15]:


model = Sequential() #Need to first initiliaze model

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(128, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) #Replace every 2,2 with a single pixel

model.add(Dropout(0.25)) #To reduce overfitting model

model.add(Flatten())

model.add(Dense(128, activation='relu')) #This is our first fully connected layer 

model.add(Dropout(0.5)) #Normally dropout after Dense layer

model.add(Dense(num_classes, activation='softmax')) #Second Dense Layer
#We want to the output to be out of 10 classes so use num_classes


# Lets see what this model looks like _👀_

# In[16]:


model.summary()


# Creating copies of the model for the different optimizers

# In[17]:


model_SGD = model
model_Adam = model
model_Adagrad = model


# Will need to reset the four variables below before we can train the next model using a different optimizer

# In[18]:


x_train.shape
x_train_reset = x_train


# In[19]:


y_train.shape
y_train_reset = y_train


# In[20]:


x_test.shape
x_test_reset = x_test


# In[21]:


y_test.shape
y_test_reset = y_test


# ## SGD Optimizer

# #### Configure the model for loss, optimization, and metrics

# In[22]:


model_SGD.compile(#Loss
              loss=keras.losses.categorical_crossentropy,
              #Optimizer
              optimizer=keras.optimizers.SGD(lr=0.10, momentum=0.01, decay=0.005),
              #Metrics
              metrics=['accuracy'])


# ### Train my data, train! 🏃

# In[23]:


get_ipython().run_cell_magic(u'time', u'', u'\nmodel_SGD.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))')


# We can see the x_test is a 4 degree tensor holding 10000 elements of 28 by 28 pictures with 1 channel that was tested. 

# In[24]:


x_test.shape


# y_test is the '10000' image results assigned to one of '10' classes

# In[25]:


y_test.shape


# Checking to see how well the model learned. How much was lost in the learning, and how accurate it was when testing.

# In[26]:


score = model_SGD.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])

print('Test accuray:', score[1])


# #### Perform Confusion Matrix 

# Imports to perform confusion matrix. sklearn will have confusion matrix function and np will have argmax function we will see later

# In[27]:


from sklearn.metrics import confusion_matrix


# In[28]:


import numpy as np


# Retrieving what the model predicted and stroing in matrix 'y_pred'

# In[29]:


y_pred = model_SGD.predict(x_test)


# In[30]:


y_pred


# For each row in the matrix, finds the highest argument out of possible 10 and returns the array position, this value is the predicted class by the model. axis=1 turns these array poisition results this into an array for each image of 10000

# y_test contains the actual values of what each image should be in a matrix with rows containing a 1 in the position of the correct class. We find this class by again finding the array position using argmax and then storing it. Axis 1 again, putting this into an array of 10000

# In[31]:


classes = np.argmax(y_pred, axis=1) #Find highest value in y_pred matrix
# Want to convert y_pred to a array of 10 classes to specify what class it is
# t-shirt, shirt, etc.
y_true = np.argmax(y_test, axis=1)#We also performed a one hot encoding on y_test and need
#to reduce it to an actual numerical value


# In[32]:


classes.shape


# In[33]:


y_true


# Confusion matrix runs the predicited classes along the coloumns of the matrix and the actual values along the rows of the matrix. The diagonal is the number of correct predictions for each. The numbers alongside the diagonal represent confusion or mistakes by the model and counts these errors. Higher numbers reperesent higher confusion between two different classes.

# In[34]:


confusion_matrix(y_true, classes)


# #### Now trying the data with other optimizers, for this one, Adam.

# ### Adam Optimizer

# In[35]:


x_train = x_train_reset
y_train = y_train_reset
x_test = x_test_reset
y_test = y_test_reset


# In[36]:


model_Adam.compile(#Loss
              loss=keras.losses.categorical_crossentropy,
              #Optimizer
              optimizer=keras.optimizers.Adam(), #leaving Adam defaults
              #Metrics
              metrics=['accuracy'])
              #for another day
              #optimizer=keras.optimizers.Adam(lr=0.10, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005)


# In[37]:


get_ipython().run_cell_magic(u'time', u'', u'\nmodel_Adam.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))')


# In[38]:


score = model_Adam.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])

print('Test accuray:', score[1])


# #### Adam Confusion Matrix

# In[39]:


y_pred = model_Adam.predict(x_test)


# In[40]:


classes = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)


# In[41]:


confusion_matrix(y_true, classes)


# ### Adagrad Optimizer

# In[42]:


x_train = x_train_reset
y_train = y_train_reset
x_test = x_test_reset
y_test = y_test_reset


# In[43]:


model_Adagrad.compile(#Loss
              loss=keras.losses.categorical_crossentropy,
              #Optimizer
              optimizer=keras.optimizers.Adagrad(#leaving at defaults),
              ),
              #Metrics
              metrics=['accuracy'])


# In[44]:


get_ipython().run_cell_magic(u'time', u'', u'\nmodel_Adagrad.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))')


# In[45]:


score = model_Adagrad.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])

print('Test accuray:', score[1])


# #### Adagrad Confusion Matrix

# In[46]:


y_pred = model_Adagrad.predict(x_test)


# In[47]:


classes = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)


# In[48]:


confusion_matrix(y_true, classes)


# ### Conclusions

# The accuracies were, SGD-87.1%, Adam-92.7%, Adagrad-93.3%

# The model seemes to confuse shirts with t-shirts, shirts with dresses, and shirts with coats. There was also some confusion between sneakers and ankle boots.

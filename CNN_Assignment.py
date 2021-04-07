import numpy as np
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

## Load is Cifar 10 dataset
(x_test, y_test), (x_train, y_train) = cifar10.load_data()
## Print ouf shapes of training and testing sets
print("train ", x_train.shape)
print("/test ", x_test.shape)
## Normalize x train and x test images
x_train = x_train/255.0
x_test = x_test/255.0
## Create one hot encoding vectors for y train and y test
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
## Define the model
model = Sequential()
## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, same padding and input shape
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu",kernel_initializer='he_uniform', padding="same", input_shape=(32,32,3)))
## Add a batch normalization layer
model.add(BatchNormalization())
## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding 
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", kernel_initializer='he_uniform', padding="same"))
## Add a batch normalization layer
model.add(BatchNormalization())
## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size=(2,2)))
## Add dropout layer of 0.2
model.add(Dropout(0.2))
## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", kernel_initializer='he_uniform', padding="same"))
## Add a batch normalization layer
model.add(BatchNormalization())
## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu",kernel_initializer='he_uniform', padding="same"))
## Add a batch normalization layer
model.add(BatchNormalization())
## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size=(2,2)))
## Add dropout layer of 0.2
model.add(Dropout(0.2))
## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer='he_uniform', padding="same"))
## Add a batch normalization layer
model.add(BatchNormalization())
## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer='he_uniform', padding="same"))
## Add a batch normalization layer
model.add(BatchNormalization())
## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size=(2,2)))
## Add dropout layer of 0.2
model.add(Dropout(0.2))
## Flatten the resulting data
model.add(Flatten())
## Add a dense layer with 128 nodes, relu activation and he uniform kernel initializer
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
## Add a batch normalization layer
model.add(BatchNormalization())
## Add dropout layer of 0.2
model.add(Dropout(0.2))
## Add a dense softmax layer
model.add(Dense(10, activation='softmax'))
## Set up early stop training with a patience of 3
callback = EarlyStopping(patience=3)
## Compile the model with adam optimizer, categorical cross entropy and accuracy metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Accuracy()])
# Image Data Generator , we are shifting image accross width and height of 0.1 also we are flipping the image horizantally and rotating the images by 20 degrees
generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, rotation_range=20)
## Take data and label arrays to generate a batch of augmented data, default parameters are fine.
newGen = generator.flow(x_train, y_train)
## Define the number of steps to take per epoch as training examples over 64
print(len(x_train))
steps=len(x_train)/64
print("steps: ", steps)
## Fit the model with the generated data, 200 epochs, steps per epoch and validation data defined. 
fitted = model.fit(newGen, epochs = 200, steps_per_epoch=steps, validation_data=(x_test, y_test))
print("Accuracy: ", fitted.history['accuracy'])

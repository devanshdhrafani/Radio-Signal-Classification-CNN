#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import warnings;warnings.simplefilter('ignore')
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from time import time
from sklearn.metrics import confusion_matrix
from sklearn import metrics

print('Tensorflow version:', tf.__version__)

train_images = pd.read_csv('dataset/train/images.csv', header=None)
train_labels = pd.read_csv('dataset/train/labels.csv', header=None)
val_images = pd.read_csv('dataset/validation/images.csv', header=None)
val_labels = pd.read_csv('dataset/validation/labels.csv', header=None)

print("Training set shape:",train_images.shape,train_labels.shape)
print("Validation set shape:",val_images.shape,val_labels.shape)

x_train = train_images.values.reshape(3200, 64, 128, 1)
x_val = val_images.values.reshape(800, 64, 128, 1 )

y_train = train_labels.values
y_val = val_labels.values

# Randomly plot any 3 spectograms to get idea of the data
plt.figure(0, figsize=(12,12))
for i in range(1,4):
    plt.subplot(1,3,i)
    img = np.squeeze(x_train[np.random.randint(0,x_train.shape[0])])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap = 'gray')

datagen_train = ImageDataGenerator(rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.4,
                                   zoom_range=[0.8, 1.2],
                                   horizontal_flip=True,
                                  vertical_flip=True,
                                   fill_mode='wrap'
                                  )
datagen_train.fit(x_train)

datagen_val = ImageDataGenerator(rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.4,
                                   zoom_range=[0.8, 1.2],
                                   horizontal_flip=True,
                                  vertical_flip=True,
                                   fill_mode='wrap')
datagen_val.fit(x_val)


# Initialising the CNN
model = Sequential()
# 1st Convolution
model.add(Conv2D(32, (5,5), padding='same', input_shape=(64, 128, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
# 2nd Convolution layer
model.add(Conv2D(64, (5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Flattening
model.add(Flatten())
# Fully connected layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(4, activation='softmax'))

initial_learning_rate = 0.005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = initial_learning_rate,
    decay_steps = 5,
    decay_rate = 0.96,
    staircase = True
)
optimizer = Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint(filepath='model_weight2.h5', monitor='val_loss', 
                             save_weights_only=True, mode='min', verbose=0)
callbacks = [TensorBoard(log_dir='C:/tensorflowLogs/{}'.format(time())), checkpoint]

batch_size = 32

history = model.fit(
    datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
    steps_per_epoch = len(x_train) // batch_size,
    validation_data = datagen_val.flow(x_val, y_val, batch_size=batch_size, shuffle=True),
    validation_steps = len(x_val) // batch_size,
    epochs = 12,
    callbacks=callbacks
)

model.evaluate(x_val, y_val)

y_true = np.argmax(y_val, 1)
y_pred = np.argmax(model.predict(x_val), 1)
print(metrics.classification_report(y_true, y_pred))

print("Classification accuracy: %0.6f" %metrics.accuracy_score(y_true, y_pred))

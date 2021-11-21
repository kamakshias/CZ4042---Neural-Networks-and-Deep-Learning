# Importing the header files

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.io import read_file
from matplotlib import image
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.regularizers import l2
import numpy as np
import pandas as pd
import time
import math
from tensorflow.image import resize
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import multiprocessing as mp
import statistics

from tensorflow.keras.callbacks import Callback
from keras.preprocessing.image import load_img
from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Lambda
from tensorflow.keras import Sequential
from keras.regularizers import l2

from tensorflow.keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/UG/kamakshi001/fold_frontal_0_data.txt', sep='\t')
df1 = pd.read_csv('/home/UG/kamakshi001/fold_frontal_1_data.txt', sep='\t')
df2 = pd.read_csv('/home/UG/kamakshi001/fold_frontal_2_data.txt', sep='\t')
df3 = pd.read_csv('/home/UG/kamakshi001/fold_frontal_3_data.txt', sep='\t')
df4 = pd.read_csv('/home/UG/kamakshi001/fold_frontal_4_data.txt', sep='\t')


# converting the string labels to numerical labels
def df_to_list(df):
    data2 = []
    for i in range(len(df)):
        path = "/home/UG/kamakshi001/aligned/" + \
            str(df['user_id'][i])+"/landmark_aligned_face." + \
            str(df['face_id'][i])+"."+str(df['original_image'][i])
        if df['gender'][i] == 'm':
            gender = 0
        else:
            gender = 1

        if str(df['age'][i]) == str((0, 2)):
            age = 0
        elif str(df['age'][i]) == str((4, 6)):
            age = 1
        elif str(df['age'][i]) == str((8, 13)):
            age = 2
        elif str(df['age'][i]) == str((15, 20)):
            age = 3
        elif str(df['age'][i]) == str((25, 32)):
            age = 4
        elif str(df['age'][i]) == str((38, 43)):
            age = 5
        elif str(df['age'][i]) == str((48, 53)):
            age = 6
        elif str(df['age'][i]) == str((60, 100)):
            age = 7
        data2.append([path, gender, age])
    return data2

# Combining all the folds
df_list1 = df_to_list(df)
df_list2 = df_to_list(df1)
df_list3 = df_to_list(df2)
df_list4 = df_to_list(df3)
df_list5 = df_to_list(df4)

df_list = []
for i in (df_list1):
    df_list.append(i)
for i in (df_list2):
    df_list.append(i)
for i in (df_list3):
    df_list.append(i)
for i in (df_list4):
    df_list.append(i)
for i in (df_list5):
    df_list.append(i)


# Creating a new dataset
df_new = pd.DataFrame(df_list, columns=['Path', 'Gender', 'Age'])
df_new.to_csv("/home/UG/kamakshi001/data_filtered.csv", index=False)

df_image_x = []
df_gender_y = []
df_age_y = []

# Loading the images
for i in range(len(df_new)):
    image = Image.open(df_new['Path'][i])
    image = np.array(image)
    image = tf.image.resize(image, [256, 256])
    data = asarray(image)
    for j in data:
        j = j/255
    df_image_x.append(data)
    df_gender_y.append(df_new['Gender'][i])
    df_age_y.append(df_new['Age'][i])

# Test, Train, Validation split
x_train, y_train, x_test, y_test, x_val, y_val = df_image_x[0:9492], df_gender_y[0:9492], df_image_x[
    9492:11527], df_gender_y[9492:11527], df_image_x[11527:13561], df_gender_y[11527:13561]

x_train = np.asarray(x_train, dtype='float32')
y_train = np.asarray(y_train, dtype='float32')
x_val = np.asarray(x_val, dtype='float32')
y_val = np.asarray(y_val, dtype='float32')
x_test = np.asarray(x_test, dtype='float32')
y_test = np.asarray(y_test, dtype='float32')


# Step Decay for adaptive learning rate
def step_decay(epoch):
    init_lrate = 1e-3  # TOCHANGE
    drop = 0.1
    epochs_drop = 10000
    lrate = init_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# Definition of weight initializers, optimizers, loss function and learning rate

weight_init = keras.initializers.TruncatedNormal(
    mean=0.0, stddev=0.01, seed=10)

#bias_init = tf.keras.initializers.Constant(value=0.1)

sgd = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)  

loss_func = keras.losses.BinaryCrossentropy()

lrate = keras.callbacks.LearningRateScheduler(step_decay)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
])

# Model Initialization

gender_model = keras.models.Sequential([
    Conv2D(96, (7, 7), input_shape=(256, 256, 3), strides=4,
           padding='valid', activation='relu', kernel_initializer=weight_init),
    data_augmentation,
    MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),
    Lambda(lambda x: tf.nn.local_response_normalization(
        input=x, alpha=0.0001, beta=0.75)),

    Conv2D(256, (5, 5), padding='same', activation='relu',
           kernel_initializer=weight_init),
    MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),
    Lambda(lambda x: tf.nn.local_response_normalization(
        input=x, alpha=0.0001, beta=0.75)),

    Conv2D(384, (3, 3), padding='same', activation='relu',
           kernel_initializer=weight_init),
    MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),
    Flatten(),

    Dense(512, activation="relu", kernel_initializer=weight_init),
    Dropout(0.5),

    Dense(512, activation='relu', kernel_initializer=weight_init),
    Dropout(0.5),

    Dense(1, activation='sigmoid', kernel_initializer=weight_init)
])

gender_model.compile(loss=loss_func, optimizer=sgd, metrics=['accuracy'])

num_epochs = 200
batch_size = 16
seed = 10
save_model = True
np.random.seed(seed)
tf.random.set_seed(seed)
drive_prefix = "/home/UG/kamakshi001/augmentation models"
checkpoint = keras.callbacks.ModelCheckpoint(
    drive_prefix + '/augmentation_checkpoint.h5', monitor='val_accuracy', verbose=1, mode='max', save_best_only=True)
csv_logger = keras.callbacks.CSVLogger(
    drive_prefix + '/augmentation_csvlog.csv')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
if save_model:
    callbacks = [lrate, checkpoint, csv_logger, early_stopping]
else:
    callbacks = [lrate, csv_logger, early_stopping]

# Train the model

results = gender_model.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=num_epochs,
                           verbose=1,
                           use_multiprocessing=True,
                           callbacks=callbacks,
                           validation_data=(x_val, y_val))

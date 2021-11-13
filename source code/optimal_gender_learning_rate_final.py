#!/usr/bin/env python
# coding: utf-8

# # Importing header files

# In[2]:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.io import read_file
from matplotlib import image
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.regularizers import l2
import numpy as np
import pandas as pd
import time
from tensorflow.image import resize
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)
from tensorflow.keras.callbacks import Callback
from keras.preprocessing.image import load_img
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from pprint import pprint


# # Experimenting with learning rates [0.01,0.001,0.005] with:
# ## 1) Number of neurons = 256
# ## 2) Batch size = 4 [optimal]
# ## 3) Number of CNN layers as 4
# 

# In[3]:


num_neurons = 256
batch_size=4
num_layers=4


# # Importing the dataset text files

# In[4]:


df=pd.read_csv('/home/UG/kamakshi001/fold_frontal_0_data.txt', sep='\t')
df1=pd.read_csv('/home/UG/kamakshi001/fold_frontal_1_data.txt', sep='\t')
df2=pd.read_csv('/home/UG/kamakshi001/fold_frontal_2_data.txt', sep='\t')
df3=pd.read_csv('/home/UG/kamakshi001/fold_frontal_3_data.txt', sep='\t')
df4=pd.read_csv('/home/UG/kamakshi001/fold_frontal_4_data.txt', sep='\t')


# # Function to pre-process the data
# ## 1) Gender =[0:Male,1:Female]
# ## 2) Age= [0:(0,2),1:(4,6),2:(8,13),3:(15,20),4:(25,32),5:(38,43),6:(48,53),7:(60,100)]

# In[57]:


def df_to_list(df):
    data2=[]
    for i in range(len(df)):
        path="/home/UG/kamakshi001/aligned/"+str(df['user_id'][i])+"/landmark_aligned_face."+str(df['face_id'][i])+"."+str(df['original_image'][i])              
        if df['gender'][i]=='m':
            gender=0
        else:
            gender=1

        if str(df['age'][i]) == str((0,2)):
            age=0
        elif str(df['age'][i]) == str((4,6)):
            age=1
        elif str(df['age'][i]) == str((8,13)):
            age=2
        elif str(df['age'][i]) == str((15,20)):
            age=3
        elif str(df['age'][i]) == str((25,32)):
            age=4
        elif str(df['age'][i]) == str((38,43)):
            age=5
        elif str(df['age'][i]) == str((48,53)):
            age=6
        elif str(df['age'][i]) == str((60,100)):
            age=7
        data2.append([path,gender,age])
    return data2


# In[58]:


df_list1=df_to_list(df)
df_list2=df_to_list(df1)
df_list3=df_to_list(df2)
df_list4=df_to_list(df3)
df_list5=df_to_list(df4)


# # Concatenating all lists to one

# In[59]:


df_list=[]
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


# # Saving the New precessed dataset into a csv file

# In[60]:


df_new = pd.DataFrame(df_list, columns = ['Path', 'Gender','Age'])
df_new.to_csv("/home/UG/kamakshi001/data_filtered.csv",index=False)
df_new=df_new.sample(len(df_new))


# # Test, train, validation divide. Also normalization of all pixels so that all values are between 0 and 1.

# In[ ]:



df_image_x=[]
df_gender_y=[]
df_age_y=[]
for i in range(len(df_new)):
    image = Image.open(df_new['Path'][i])
    image = np.array(image)
    image = tf.image.resize(image, [256, 256]) 
    data = asarray(image)
    for j in data:
        j=j/255
    df_image_x.append(data)
    df_gender_y.append(df_new['Gender'][i])
    df_age_y.append(df_new['Age'][i])


# In[52]:




x_train, y_train, x_val, y_val, x_test, y_test = df_image_x[0:9492],df_gender_y[0:9492],df_image_x[9492:11527],df_gender_y[9492:11527],df_image_x[11527:13561],df_gender_y[11527:13561]

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_val = np.asarray(x_val)
y_val = np.asarray(y_val)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)




# # Model Definition

# # Strating Training with different batch sizes

# In[ ]:

#Definition of weight initializers, optimizers, loss function and learning rate


# # Model Definition

# # Strating Training with different neuron numbers

# In[ ]:


learning_rates=[0.01,0.001,0.005]
for learning_rate in learning_rates:
    def step_decay(epoch):
        init_lrate = learning_rate #TOCHANGE
        drop = 0.1
        epochs_drop = 10000
        lrate = init_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    sgd = keras.optimizers.SGD(learning_rate,momentum=0.0) #TOCHANGE
    weight_init = keras.initializers.TruncatedNormal(mean=0.0,stddev=0.01,seed=10)
    loss_func = keras.losses.BinaryCrossentropy()
    lrate = keras.callbacks.LearningRateScheduler(step_decay)

    gender_model = keras.models.Sequential([    
        Conv2D(96, (7,7), input_shape=(256,256,3), strides=4, padding='valid', activation='relu', kernel_initializer = weight_init),
        MaxPooling2D(pool_size = (3, 3), strides = 2, padding='same'),
        Lambda(lambda x: tf.nn.local_response_normalization(input=x, alpha=0.0001, beta=0.75)),

        Conv2D(256, (5,5), padding='same', activation = 'relu', kernel_initializer = weight_init),
        MaxPooling2D(pool_size = (3, 3), strides = 2, padding = 'same'),
        Lambda(lambda x: tf.nn.local_response_normalization(input=x, alpha=0.0001, beta=0.75)),

        Conv2D(384,(3,3), padding='same', activation='relu', kernel_initializer = weight_init),
        MaxPooling2D(pool_size=(3,3), strides = 2, padding='same'),
        Flatten(),

        Dense(512, activation = "relu", kernel_initializer = weight_init),
        Dropout(0.5),

        Dense(512, activation='relu',kernel_initializer = weight_init),
        Dropout(0.5),

        Dense(1, activation ='sigmoid', kernel_initializer = weight_init)
    ])
    
    gender_model.compile(loss = loss_func, optimizer = sgd, metrics=['accuracy'])
    checkpoint_filepath = "/home/UG/kamakshi001/best_model_gender_"+str(batch_size)+"_"+str(learning_rate)+"_"+str(num_neurons)+"_"+str(num_layers)+".tf"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
    early_stopping=EarlyStopping(monitor='val_loss',patience=10)
    history = age_model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=140,
                        verbose=1,
                        use_multiprocessing=True,
                        callbacks=[early_stopping,model_checkpoint_callback],
                        validation_data=(x_val, y_val))
    loss=[]
    accuracy=[]
    val_loss=[]
    val_accuracy=[]
    loss=history.history['loss'].copy()
    val_loss=history.history['val_loss'].copy()
    val_accuracy=history.history['val_accuracy'].copy()
    accuracy=history.history['accuracy'].copy()
    path_1="/home/UG/kamakshi001/data_eval_age_final_"+str(batch_size)+"_"+str(learning_rate)+"_"+str(num_neurons)+"_"+str(num_layers)+".csv"
    df_eval=pd.DataFrame(list(zip(loss,val_loss,val_accuracy,accuracy)),columns=['loss','val_loss','val_accuracy','accuracy'])
    df_eval.to_csv(path_1 ,index=False)
    path_2="/home/UG/kamakshi001/data_age_scores_final_"+str(batch_size)+"_"+str(learning_rate)+"_"+str(num_neurons)+"_"+str(num_layers)+".csv"
    age_model.load_weights(checkpoint_filepath)
    scores = age_model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
    score_1=[]
    score_1.append(scores)
    df_score=pd.DataFrame(score_1,columns=['loss','accuracy'])
    df_score.to_csv(path_2 ,index=False)



# In[ ]:

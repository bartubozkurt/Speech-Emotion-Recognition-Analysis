#!/usr/bin/env python
# coding: utf-8

# ## Bartu Bozkurt - Bilgisayar Bilimleri - 2017280013

# In[ ]:


# Requiremnts Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
import os.path
import os
import IPython.display
import seaborn as sns
import librosa
import librosa.display
import soundfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D,Flatten, BatchNormalization
from keras import optimizers
import warnings


# In[ ]:



image_dir = Path('/kaggle/input/datademo/demodata')
filepaths = list(image_dir.glob(r'**/*.wav'))
labels = list(map(lambda x : os.path.split(x)[1].split('_')[2],filepaths))
set(labels)
        
# ANG: Anger
# DIS: Disgust
# FEA: Fear 
# HAP: Happiness
# NEU: Neutral 
# SAD: Sadness


# In[ ]:


filepaths = pd.Series(filepaths,name = 'Filepath').astype(str)
labels = pd.Series(labels,name = 'Label')

#dataframe !!!
audio_DataFrame = pd.concat([filepaths,labels], axis = 1)
audio_DataFrame.head(15)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use("ggplot")


# In[ ]:


plt.title('COUNT OF EMOTIONS')
sns.countplot(x=audio_DataFrame["Label"])
sns.despine(top=True, right=True, left=False, bottom=False)

# ANG: Anger || DIS: Disgust || FEA: Fear || HAP: Happiness || NEU: Neutral || SAD: Sadness


# ## Load Data

# In[ ]:


audio_arr = []

for i in audio_DataFrame["Filepath"]:
    x,sr= librosa.load(i,sr=44100)
    audio_arr.append(x)
    
audio_DataFrame["Arrays"] = audio_arr


# In[ ]:


audio_DataFrame


# ### ANGER

# In[ ]:


angry_file = audio_DataFrame[audio_DataFrame['Label'] == 'ANG']['Filepath']
angry_arr = audio_DataFrame[audio_DataFrame['Label'] ==  'ANG']['Arrays']

librosa.display.waveplot(angry_arr.iloc[0], color='#C00808')
IPython.display.Audio(angry_file.iloc[0])


# ### DISGUST

# In[ ]:


disgust_file = audio_DataFrame[audio_DataFrame['Label'] == 'DIS']['Filepath']
disgust_array = audio_DataFrame[audio_DataFrame['Label'] == 'DIS']['Arrays']

librosa.display.waveplot(disgust_array.iloc[0], color='#804E2D')
IPython.display.Audio(disgust_file.iloc[0])


# ### FEAR

# In[ ]:


fear_file = audio_DataFrame[audio_DataFrame['Label'] == 'FEA']['Filepath']
fear_array = audio_DataFrame[audio_DataFrame['Label'] == 'FEA']['Arrays']

librosa.display.waveplot(fear_array.iloc[0], color='#7D55AA')
IPython.display.Audio(fear_file.iloc[0])


# ### HAPPINESS

# In[ ]:


happy_file = audio_DataFrame[audio_DataFrame['Label'] == 'HAP']['Filepath']
happy_array = audio_DataFrame[audio_DataFrame['Label'] == 'HAP']['Arrays']

librosa.display.waveplot(happy_array.iloc[0], color='#F19C0E')
IPython.display.Audio(happy_file.iloc[0])


# ### NEUTRAL

# In[ ]:


neu_file = audio_DataFrame[audio_DataFrame['Label'] == 'NEU']['Filepath']
neu_array = audio_DataFrame[audio_DataFrame['Label'] == 'NEU']['Arrays']

librosa.display.waveplot(neu_array.iloc[0], color='#4CB847')
IPython.display.Audio(neu_file.iloc[0])


# ### SADNESS

# In[ ]:


sad_file = audio_DataFrame[audio_DataFrame['Label'] == 'SAD']['Filepath']
sad_array = audio_DataFrame[audio_DataFrame['Label'] == 'SAD']['Arrays']

librosa.display.waveplot(sad_array.iloc[0], color='#478FB8')
IPython.display.Audio(sad_file.iloc[0])


# ## Data Augmentation

# In[ ]:


def noise(data):
    noise = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise * np.random.normal(size = data.shape[0])
    return data

def stretch(data, rate = 0.8):
    return librosa.effects.time_stretch(data,rate)

def pitch(data, sampling_rate, pitch_factor = 0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


# ## Extracting Features

# ![](https://www.researchgate.net/profile/Buket-Barkana/publication/259823741/figure/fig2/AS:299389261238276@1448391370399/Definition-of-zero-crossings-rate.png)

# In[ ]:


def extract_features(data):
    # Zero Crossing Rate
    result = np.array([])  
    zcr = np.mean(librosa.feature.zero_crossing_rate(y = data).T, axis = 0)
    result = np.hstack((result,zcr))
    
    # Croma_stft
    stft = np.array(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S = stft, sr = sr, n_fft = 200).T, axis = 0)
    result = np.hstack((result,chroma_stft))
    
    # Mfcc
    mfcc = np.mean(librosa.feature.mfcc(y=data,sr = sr, n_fft = 200).T, axis =0)
    result = np.hstack((result,mfcc))
    
    # Melspectogram
    mel = np.mean(librosa.feature.melspectrogram(y= data, sr = sr, n_fft = 200).T, axis = 0)
    result = np.hstack((result,mel))
    
    # Tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y = data, sr = sr).T,axis = 0)
    result = np.hstack((result,tonnetz))
    
    return result


# In[ ]:


def get_features(data):
    
    result = []
    
    result1=  extract_features(data)
    result.append(result1)
    
    noise_data = noise(data)
    result2=  extract_features(noise_data)
    result.append(result2)
    
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data,sr)
    result3=  extract_features(data_stretch_pitch)
    result.append(result3)
    
    return result


# In[ ]:


warnings.filterwarnings('ignore')

x = []
y = []

for i in range(len(audio_DataFrame)):
    feature = get_features(audio_DataFrame['Arrays'].iloc[i])
    for j in feature:
        x.append(j)
        y.append(audio_DataFrame['Label'].iloc[i])


# In[ ]:


labelencoder = LabelEncoder()
y = utils.to_categorical(labelencoder.fit_transform(y))

y


# ## Train Test Split

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.1)
print((x_train.shape, y_train.shape, x_test.shape, y_test.shape))


# In[ ]:


x_train = np.expand_dims(x_train, axis = 2)
x_test = np.expand_dims(x_test, axis = 2)


# In[ ]:


print((x_train.shape, y_train.shape, x_test.shape, y_test.shape))


# ## Creating Model

# In[ ]:


model = Sequential()

model.add(Conv1D(128,3,activation = 'relu', input_shape = (x_train.shape[1],1)))
model.add(MaxPooling1D((1)))

model.add(Conv1D(256,3,activation = 'relu'))
model.add(MaxPooling1D((1)))

model.add(Conv1D(512, 3, activation='relu'))
model.add(MaxPooling1D((1)))

model.add(Conv1D(1024, 3, activation='relu'))
model.add(MaxPooling1D((1)))

model.add(Flatten())

model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(6,activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.RMSprop(lr = 0.0005),
              metrics = ['Accuracy'])

model.summary()


# In[ ]:


history = model.fit(x_train,y_train,
                   epochs = 50,
                   batch_size = 128,
                   validation_data = (x_test,y_test))


# In[ ]:


y_prediction = model.predict(x_test)
matrix = confusion_matrix(y_test.argmax(axis = 1), 
                          y_prediction.argmax(axis = 1))

ax = sns.heatmap(matrix, annot  = True, fmt = 'd', cmap = 'rocket_r',
                 xticklabels  = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness'],
                 yticklabels = ['Anger' , 'Disgust' , 'Fear', 'Happiness' , 'Neutral' , 'Sadness'])


# ##### Bartu Bozkurt - May 2021

# In[ ]:





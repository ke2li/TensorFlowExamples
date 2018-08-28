#Recurrent neural network trained to generate Donald Trump tweets
#Tweets taken from trumptwitterarchive.com/archive
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dropout, Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#data = pd.read_csv("RawTweets.csv", index_col=False, dtype={'text': str})

#character level implementation
data = []
with open("RawTweets.txt", 'r', encoding='utf-8') as f:
	for line in f:
		data.append(list(line.encode(utf-8)))

data = keras.preprocessing.sequence.pad_sequences(data, maxlen=140, dtype='int32', padding='post', truncating='post', value='0')

model = keras.Sequence()
model.add(LSTM(512))
model.add(Dropout(0.5))
model.add(LSTM(512))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy')
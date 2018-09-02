#Recurrent neural network trained to generate Donald Trump tweets
#Tweets taken from trumptwitterarchive.com/archive
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
import sys
import math
import string
#data = pd.read_csv("RawTweets.csv", index_col=False, dtype={'text': str})

#global variables
SEQ_LENGTH = 80
HIDDEN_DIM = 512
LAYER_NUM = 3

def CreateModel():
	model = keras.Sequential()
	model.add(keras.layers.LSTM(HIDDEN_DIM, input_shape=(None, num_vocab), return_sequences=True))
	model.add(keras.layers.Dropout(0.5))
	for i in range(LAYER_NUM -1):	
		model.add(keras.layers.LSTM(HIDDEN_DIM, return_sequences=True))
		model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_vocab)))
	model.add(keras.layers.Activation('softmax'))
	return model

def LoadModel():
	return keras.models.load_model('.\\ModelWeights\\weights.10-1.50.hdf5')

class Predictions(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.predictions = []
	def on_train_end(self, logs={}):
		with open("sampleTweets.txt", "a") as f:
			for tweet in self.predictions:
				f.write(tweet)
	def on_epoch_end(self, epoch, logs={}):
		ix = [np.random.randint(num_vocab)]
		y_pred = [ix_to_char[ix[-1]]]
		x = np.zeros((1, 140, num_vocab))
		for i in range(140):
			x[0,i,:][ix[-1]] = 1
			ix = np.argmax(self.model.predict(x[:, :i+1, :])[0],1)
			y_pred.append(ix_to_char[ix[-1]])

		self.predictions.append(''.join(y_pred))
		return

def ProcessData():
	#character level implementation
	file = open("RawTweets.txt", 'r', encoding='utf-8')
	lines = file.read()
	#remove URLs
	#lines = re.sub(r'^http?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
	data = lines.replace("\n", "")

	#filtering out non-printable characters
	printable = set(string.printable)
	filtered = ''.join(filter(lambda x: x in printable,data))
	data = filtered

	chars = list(set(data))
	num_chars = len(data)
	num_vocab = len(chars)
	ix_to_char = {ix:char for ix, char in enumerate(chars)}
	char_to_ix = {char:ix for ix, char in enumerate(chars)}
	num_seq = math.floor(num_chars/SEQ_LENGTH)
	print(num_seq, num_chars, num_vocab)

	x = np.zeros((num_seq, SEQ_LENGTH, num_vocab))
	y = np.zeros((num_seq, SEQ_LENGTH, num_vocab))
	for i in range(num_seq):
		x_seq = data[i*SEQ_LENGTH : (i+1)*SEQ_LENGTH]
		y_seq = data[(i*SEQ_LENGTH)+1 : ((i+1)*SEQ_LENGTH)+1]

		#vectorize
		vec_x_seq = [char_to_ix[x] for x in x_seq]
		vec_y_seq = [char_to_ix[y] for y in y_seq]

		#vocab features, onehot encode
		x_input_seq = np.zeros((SEQ_LENGTH, num_vocab))
		y_input_seq = np.zeros((SEQ_LENGTH, num_vocab))
		for j in range(SEQ_LENGTH):
			x_input_seq[j][vec_x_seq[j]] = 1.
			y_input_seq[j][vec_y_seq[j]] = 1.

		x[i] = x_input_seq
		y[i] = y_input_seq

	#split for validation data
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	np.save('train_data.npy', (x_train, y_train))
	np.save('test_data.npy', (x_test, y_test))
	return (x_train, y_train), (x_test, y_test)

def LoadData():
	(x_train, y_train), (x_test, y_test) = np.load('train_data.npy'), np.load('test_data.npy')
	return (x_train, y_train), (x_test, y_test)

#model = CreateModel()
#(x_train, y_train), (x_test, y_test) = ProcessData()
(x_train, y_train), (x_test, y_test) = LoadData()
model = LoadModel()

#callbacks
save_model = keras.callbacks.ModelCheckpoint(".\\ModelWeights\\weights.{epoch:02d}-{val_loss:.2f}.hdf5")
preds = Predictions()
earlyStop = keras.callbacks.EarlyStopping(min_delta=0.001, patience=2)

#training
model.compile(optimizer=keras.optimizers.RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test), callbacks=[earlyStop, preds, save_model])
test_loss, test_acc = model.evaluate(x_test,y_test)

# print('Test accuracy:', test_acc)

# plt.figure(figsize=[8,6])
# plt.plot(history.history['loss'],'r',linewidth=3.0)
# plt.plot(history.history['val_loss'],'b',linewidth=3.0)
# plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
# plt.xlabel('Epochs ',fontsize=16)
# plt.ylabel('Loss',fontsize=16)
# plt.title('Loss Curves',fontsize=16)

# plt.show()
 
# # Accuracy Curves
# plt.figure(figsize=[8,6])
# plt.plot(history.history['acc'],'r',linewidth=3.0)
# plt.plot(history.history['val_acc'],'b',linewidth=3.0)
# plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
# plt.xlabel('Epochs ',fontsize=16)
# plt.ylabel('Accuracy',fontsize=16)
# plt.title('Accuracy Curves',fontsize=16)

# plt.show()
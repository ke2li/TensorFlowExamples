import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

imdb= keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000, skip_top=20)
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

def decode_review(text):
	return ' '.join([reverse_word_index.get(i, '?') for i in text])

def main():
	train_data = keras.preprocessing.sequence.pad_sequences(x_train, value=word_index["<PAD>"], padding='post', maxlen=256)
	test_data = keras.preprocessing.sequence.pad_sequences(x_test, value=word_index["<PAD>"], padding='post', maxlen=256)

	model = keras.Sequential()
	model.add(keras.layers.Embedding(10000, 32, input_length=256))
	model.add(keras.layers.GlobalAveragePooling1D())
	#model.add(keras.layers.Flatten())
	#model.add(keras.layers.Dense(512, activation='relu'))
	#model.add(keras.layers.Dropout(0.25))
	model.add(keras.layers.Dense(128, activation='relu'))
	model.add(keras.layers.Dense(1, activation='sigmoid'))

	model.summary()

	model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

	history = model.fit(train_data, y_train, epochs=30, batch_size=512, validation_data=(test_data, y_test))

	test_loss, test_acc = model.evaluate(test_data,y_test)

	print('Test accuracy:', test_acc)

	plt.figure(figsize=[8,6])
	plt.plot(history.history['loss'],'r',linewidth=3.0)
	plt.plot(history.history['val_loss'],'b',linewidth=3.0)
	plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
	plt.xlabel('Epochs ',fontsize=16)
	plt.ylabel('Loss',fontsize=16)
	plt.title('Loss Curves',fontsize=16)
	
	plt.show()

	# Accuracy Curves
	plt.figure(figsize=[8,6])
	plt.plot(history.history['acc'],'r',linewidth=3.0)
	plt.plot(history.history['val_acc'],'b',linewidth=3.0)
	plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
	plt.xlabel('Epochs ',fontsize=16)
	plt.ylabel('Accuracy',fontsize=16)
	plt.title('Accuracy Curves',fontsize=16)
	plt.show()

if __name__ == '__main__':
	main()
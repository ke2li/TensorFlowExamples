import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

reuters = keras.datasets.reuters
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1020, skip_top=20)

for i in range(x_train.shape[0]):
	x_train[i] = keras.utils.to_categorical(x_train[i])

for i in range(x_test.shape[0]):
	x_test[i] = keras.utils.to_categorical(x_train[i])
y_train = keras.utils.to_categorical(y_train)
y_test= keras.utils.to_categorical(y_test)
word_index = reuters.get_word_index()

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

def decode_review(text):
	return ' '.join([reverse_word_index.get(i, '?') for i in text])

def main():
	model = keras.Sequential()
	model.add(keras.layers.Dense(512, activation='relu'))
	#model.add(keras.layers.Embedding(10000, 32))
	#model.add(keras.layers.GlobalAveragePooling1D())
	#model.add(keras.layers.Flatten())
	#model.add(keras.layers.Dense(512, activation='relu'))
	#model.add(keras.layers.Dropout(0.25))
	model.add(keras.layers.Dense(128, activation='relu'))
	model.add(keras.layers.Dense(46, activation='sigmoid'))

	#model.summary()

	model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	history = model.fit(x_train, y_train, epochs=30, batch_size=512, validation_data=(x_test, y_test))

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
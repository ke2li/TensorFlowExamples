import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import keras_metrics

reuters = keras.datasets.reuters
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

# onehot_x_train = tf.Variable(tf.zeros([x_train.shape[0], 1020]), tf.int32)
# for i in range(x_train.shape[0]):
# 	for j in x_train[i]:
# 		onehot_x_train= onehot_x_train[i,j].assign(1)

# onehot_x_test = tf.Variable(tf.zeroes([x_test.shape[0], 1020]), tf.int32)
# for i in range(x_test.shape[0]):
# 	for j in x_test[i]:
# 		onehot_x_test = onehot_x_test[i,j].assign(1)

# x_train = tf.convert_to_tensor(onehot_x_train, dtype=tf.int32)
# x_test = tf.convert_to_tensor(onehot_x_test, dtype=tf.int32)
# y_train = keras.utils.to_categorical(y_train)
# y_test= keras.utils.to_categorical(y_test)
word_index = reuters.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

train_data = keras.preprocessing.sequence.pad_sequences(x_train, value=word_index["<PAD>"], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(x_test, value=word_index["<PAD>"], padding='post', maxlen=256)
print(train_data.shape)
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

def decode_review(text):
	return ' '.join([reverse_word_index.get(i, '?') for i in text])

def main():
	model = keras.Sequential()
	#model.add(keras.layers.Dense(512, activation='relu'))
	model.add(keras.layers.Embedding(10000, 128, input_length=256))
	model.add(keras.layers.MaxPooling1D())
	model.add(keras.layers.Conv1D(64, 2, padding='same', activation='relu'))
	model.add(keras.layers.MaxPooling1D())
	model.add(keras.layers.Conv1D(32, 2, padding='same', activation='relu'))
	model.add(keras.layers.MaxPooling1D())
	#model.add(keras.layers.Flatten())
	#model.add(keras.layers.Dropout(0.4))
	#model.add(keras.layers.Embedding(1000, 32))
	#model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Flatten())
	#model.add(keras.layers.Dropout(0.4))
	#model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
	#model.add(keras.layers.Dropout(0.4))
	model.add(keras.layers.Dense(256, activation='relu'))
	#model.add(keras.layers.Dropout(0.4))
	model.add(keras.layers.Dense(128, activation='relu'))
	model.add(keras.layers.Dense(46, activation='softmax'))

	model.summary()

	earlyStop = keras.callbacks.EarlyStopping(min_delta=0.001, patience=1)

	model.compile(optimizer=keras.optimizers.Adam(0.03), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	history = model.fit(train_data, y_train, epochs=20, batch_size=256, validation_data=(test_data, y_test),callbacks=[earlyStop])

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
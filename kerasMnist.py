import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = keras.models.Sequential({
	keras.layers.Flatten(),
	keras.layers.Dense(512, activation='relu'),
	#keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(10, activation='softmax')
})

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 5, batch_size=25)

test_loss, test_acc = model.evaluate(x_test,y_test, batch_size=25)

print('Test accuracy:', test_acc)
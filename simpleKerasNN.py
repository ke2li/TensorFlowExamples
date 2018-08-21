import numpy as np
import tensorflow as tf
from tensorflow import keras


data = np.random.random((2000,32))
#labels= np.random.random((2000,10))
labels = data *5+3

val_data = np.random.random((100,32))
#val_labels = np.random.random((100,10))
val_labels = val_data+3

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64,bias_regularizer=keras.regularizers.l1(0.01)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32))

model.compile(optimizer=tf.train.GradientDescentOptimizer(0.03),
		      loss='mse',
		      metrics=['mae'])

model.fit(data, labels, epochs=100, batch_size=32,
		validation_data=(val_data, val_labels))

x = np.full((5,32), 0.5)
predictions = model.predict(x, batch_size=32)

print(predictions)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = keras.models.Sequential({
	keras.layers.Flatten(),
	keras.layers.Dense(512, activation='relu'),
	keras.layers.Dropout(0.25),
	keras.layers.Dense(128, activation='relu'),
	#keras.layers.Dropout(0.2),
	#keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(10, activation='softmax')
})

earlyStop = keras.callbacks.EarlyStopping(min_delta=0.001, patience=2)

model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[earlyStop])

test_loss, test_acc = model.evaluate(x_test,y_test)

#model.save('C:\\TensorFlow\\nnMnist.h5')

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
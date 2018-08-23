import tensorflow as tf
import tensorflow.feature_column as fc
import numpy as np
from tensorflow import keras

import os
import sys
import matplotlib.pyplot as plt

import pandas as pd

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)


def main(argv):
	boston_housing = keras.datasets.boston_housing
	(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

	column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
	                'TAX', 'PTRATIO', 'B', 'LSTAT']

	df = pd.DataFrame(data=x_train, columns=column_names)
	
	mean = x_train.mean(axis=0)
	std = x_train.std(axis=0)
	x_train = (x_train-mean) / std
	x_test = (x_test-mean) / std

	#train_input_fn = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=32, shuffle=True, num_epochs=None) 
	#test_input_fn = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=32, shuffle=False, num_epochs=1)
	feature_columns = []
	x_feature = {}
	test_feature = {}
	for label in column_names:
		feature_columns.append(tf.feature_column.numeric_column(key=label))

	for i in range(x_train.shape[1]):
		x_feature[column_names[i]] = x_train[:,i]
		test_feature[column_names[i]] = x_test[:,i]

	classifier = tf.estimator.LinearRegressor(feature_columns=feature_columns)
	classifier.train(input_fn=lambda:train_input_fn(x_feature,y_train,32), steps=5000)
	#classifier.train(input_fn=train_input_fn, steps=1000)

	eval_result = classifier.evaluate(input_fn=lambda:train_input_fn(test_feature,y_test,32))

	print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
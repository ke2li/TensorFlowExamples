# k-Nearest Neighbor
#----------------------------------
#
# This function illustrates how to use
# k-nearest neighbors in tensorflow
#
# We will use the 1970s Boston housing dataset
# which is available through the UCI
# ML data repository.
#
# Data:
#----------x-values-----------
# CRIM   : per capita crime rate by town
# ZN     : prop. of res. land zones
# INDUS  : prop. of non-retail business acres
# CHAS   : Charles river dummy variable
# NOX    : nitrix oxides concentration / 10 M
# RM     : Avg. # of rooms per building
# AGE    : prop. of buildings built prior to 1940
# DIS    : Weighted distances to employment centers
# RAD    : Index of radian highway access
# TAX    : Full tax rate value per $10k
# PTRATIO: Pupil/Teacher ratio by town
# B      : 1000*(Bk-0.63)^2, Bk=prop. of blacks
# LSTAT  : % lower status of pop
#------------y-value-----------
# MEDV   : Median Value of homes in $1,000's

# importing required libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
from tensorflow.python.framework import ops
from tensorflow.python import debug as tf_debug
# clear the default graph
ops.reset_default_graph()

# creating the tensorflow session to run operations
sess = tf.Session()
#debug wrapper
sess = tf_debug.LocalCLIDebugWrapperSession(sess)

# url to load data from
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
# all features, with final row representing y values
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# features to be used
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
num_features = len(cols_used)
# download data
housing_file = requests.get(housing_url)
# convert the data from newline and spaces formatting, while ensuring that empty lines do not get added
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]

# extracting the y values from the last column
y_vals = np.transpose([np.array([y[13] for y in housing_data])])
# extracting the data points of features to be used
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

## Min-Max Scaling to normalize data
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

# create random results that are still reproducible
np.random.seed(13)
# row indices for data points that will be in the training set
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
# row indices for data points in the test set
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
# creating the appropriate matrices
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare k-value and batch size
k = 4
batch_size=len(x_vals_test)

# Placeholders to configure model before adding real data
x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Use broadcasting to create a matrix of (# test points, distances between features, features), then sum along features
# L1
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), axis=2)
# distance is matrix of sums of features between every test point and train point
# L2
#distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=1))
# similar to L1, but values are squared before adding, then sqrt
# Predict: Get min distance index (Nearest neighbor)
#prediction = tf.arg_min(distance, 0)
# find values and indices for the k closest training points for every data point
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
# sum up distances from k nearest points
x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1),1)
#create a matrix of sums for k nearest points
x_sums_repeated = tf.matmul(x_sums,tf.ones([1, k], tf.float32))
# initialize weights to a small value that is close to the sum
x_val_weights = tf.expand_dims(tf.div(top_k_xvals,x_sums_repeated), 1)

# find corresponding y values for the k nearest points
top_k_yvals = tf.gather(y_target_train, top_k_indices)
# multiply weights by corresponding labels to obtain a simple prediction
prediction = tf.squeeze(tf.matmul(x_val_weights,top_k_yvals), axis=[1])

# Calculate mean squared error
mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)

# Calculate how many loops over training data
num_loops = int(np.ceil(len(x_vals_test)/batch_size))

for i in range(num_loops):
	# lower range of batch
    min_index = i*batch_size
    # higher range of batch, without going over total length of data
    max_index = min((i+1)*batch_size,len(x_vals_train))
    # create the data set using the indexes
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    # run prediction calculations based on data
    predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                         y_target_train: y_vals_train, y_target_test: y_batch})
    #calculate error in each run
    batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                         y_target_train: y_vals_train, y_target_test: y_batch})

    print('Batch #' + str(i+1) + ' MSE: ' + str(np.round(batch_mse,3)))

# Plot prediction and actual distribution
# creates 45 bins of size 1
bins = np.linspace(5, 50, 45)

#plot the prediction on a histogram into the bins
plt.hist(predictions, bins, alpha=0.5, label='Prediction')
#plot actual values onto the same graph 
plt.hist(y_batch, bins, alpha=0.5, label='Actual')
#insert title
plt.title('Histogram of Predicted and Actual Values')
# insert x label
plt.xlabel('Med Home Value in $1,000s')
# insert y label
plt.ylabel('Frequency')
# insert legend in upper right
plt.legend(loc='upper right')
# show the graph after constructing
plt.show()


# import required modules
import tensorflow as tf
import time

# placeholders for input data
x_ = tf.placeholder(tf.float32, shape=[4,2], name = 'x-input')
y_ = tf.placeholder(tf.float32, shape=[4,1], name = 'y-input')

# randomly initialize weights
Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name = "Theta1")
Theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name = "Theta2")

# initialize bias to 0
Bias1 = tf.Variable(tf.zeros([2]), name = "Bias1")
Bias2 = tf.Variable(tf.zeros([1]), name = "Bias2")

# perform hidden layer calcution of input * weights1 + bias1
with tf.name_scope("layer2") as scope:
	A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)

# perform output layer calculation of hidden output * weights2 + bias2
with tf.name_scope("layer3") as scope:
	Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

# calculate cost using logistic regression cost function
with tf.name_scope("cost") as scope:
	cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + 
		((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)

# other cost functions suggested by Claude Coulombe

#cost = tf.reduce_mean(tf.squared_difference(Hypothesis, y_)) 	# find difference, square then find mean

# For better result with binary classifier, use cross entropy with a sigmoid
#    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=Hypothesis, labels=y_)		#built in cost function for binary classification

# A naïve direct implementation of the loss function
#     n_instances = x_.get_shape().as_list()[0]
#     cost = tf.reduce_sum(tf.pow(Hypothesis - y_, 2))/ n_instances	# similar to first suggested cost function

# In case of problem with gradient (exploding or vanishing gradient)perform gradient clipping
#     n_instances = X.get_shape().as_list()[0]
#     cost = tf.reduce_sum(tf.pow(tf.clip_by_value(Hypothesis,1e-10,1.0) - y_,2))/(n_instances)	# includes clipping to prevent too small or too large jumps


with tf.name_scope("train") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)	#using gradient descent with 0.01 learning rate to train

XOR_X = [[0,0],[0,1],[1,0],[1,1]]	# truth table x values
XOR_Y = [[0],[1],[1],[0]]			# truth table y values

init = tf.global_variables_initializer()	#initialize variables
sess = tf.Session()	# start session

writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)	# save a graph of the model

sess.run(init)

t_start = time.clock()		#record time taken to train
for i in range(100000):
	sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
	if i % 1000 == 0:		#every 1000 iterations, print predictions, weights, biases, and cost
		print('Epoch ', i)
		print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
		print('Theta1 ', sess.run(Theta1))
		print('Bias1 ', sess.run(Bias1))
		print('Theta2 ', sess.run(Theta2))
		print('Bias2 ', sess.run(Bias2))
		print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
t_end = time.clock()
print('Elapsed time ', t_end - t_start)			#total time to train


# formatted version, every 10000 steps
# for epoch in range(100001):
#     sess.run(train_step, feed_dict={x_:XOR_X, y_: XOR_Y})
#     if epoch % 10000 == 0:
#         print("_"*80)
#         print('Epoch: ', epoch)
#         print('   Hypothesis: ')
#         for element in sess.run(Hypothesis, feed_dict={x_:XOR_X, y_: XOR_Y}):
#             print('    ',element)
#         print('   Theta1: ')
#         for element in sess.run(Theta1):
#             print('    ',element)
#         print('   Bias1: ')
#         for element in sess.run(Bias1):
#             print('    ',element)
#         print('   Theta2: ')
#         for element in sess.run(Theta2):
#             print('    ',element)
#         print('   Bias2 ')
#         for element in sess.run(Bias2):
#             print('    ',element)
#         print('   cost: ', sess.run(cost, feed_dict={x_:XOR_X, y_: XOR_Y}))
# t_end = time.clock()
# print("_"*80)
# print('Elapsed time ', t_end - t_start)

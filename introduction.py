import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[None,2])

a = tf.constant([[1,2], [3,4], [5,6]], dtype=tf.float32)
b = tf.constant([[3], [7], [11]], dtype=tf.float32)


print(a.shape)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(a)
loss = tf.losses.mean_squared_error(labels=b, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.03)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


for i in range(400):
	loss_value = sess.run((train,loss))
	print(loss_value)

print(sess.run(y_pred))

test_a = tf.constant([[1,5], [2,3], [1000,6]], dtype=tf.float32)
y_test = linear_model(test_a)
print(sess.run(y_test))
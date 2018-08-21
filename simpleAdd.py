import tensorflow as tf

a = tf.constant(5)
b = tf.constant(6)
total = a*b

print(a)
print(b)

sess = tf.Session()
print(sess.run(total))
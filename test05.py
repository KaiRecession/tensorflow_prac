import tensorflow as tf


a = zip([1, 2], [3])
for (m, n) in a:
    print(m, n)
x = []
for i in range(1):
    x.append(1.2)
x = tf.convert_to_tensor(x, dtype=float)
x = tf.reshape(x, (-1, 1))
w12 = tf.Variable(tf.constant([2.], shape=(1, 1)))
w1 = tf.constant(2.)
b1 = x - tf.constant(0.2)
w2 = tf.constant(2.)
b2 = tf.constant(1.)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
ss = tf.matmul(x, w12)
def f():
    y1 = tf.matmul(x, w12) + b1
    return y1
for i in range(1):
    optimizer.minimize(f, w12)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
y1 = 0
y3 = 0
for i in range(3):
    with tf.GradientTape(persistent=True) as tape:
        # tape.watch([w1, b1, w2, b2, w12])
        y1 += tf.matmul(x, w12) + b1
        # y3 += 2 * x * w12 + b1
        # y1 = tf.reshape(y1, (1, 1))
        # y3 = tf.reshape(y3, (1, 1))
        # y4 = tf.concat([y1, y3], axis=0)
        # y2 = y1 * w2 + b2
        # y = (y1 + y2) / 2.
    # grads里面只能够放tf的变量，常量不行
    grads = tape.gradient(y1, [w12, w12])
    # grads2 = tape.gradient(y3, [w12])
    # grads2 = tf.concat([grads, grads], axis=0)
    # grads2 = tf.distribute.get_replica_context().all_reduce('sum', grads2)
    w22 = tf.concat([w1, w1], axis=0)
    optimizer.apply_gradients((zip(grads, [w12, w12])), experimental_aggregate_gradients=False)
    print(111)
# dy2_dy1 = tape.gradient(y2, [y1])[0]
# dy1_dw1 = tape.gradient(y1, [w1])[0]
# dy2_dw1 = tape.gradient(y2, [w1])[0]
#
# print(dy2_dy1 * dy1_dw1)
# print(dy2_dw1)
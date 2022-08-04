import tensorflow as tf
import numpy as np


# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.2 + 0.3


Weights = tf.Variable(tf.random.uniform((1,), -1.0, 1.0))
biases = tf.Variable(tf.zeros((1,)))


y = Weights * x_data + biases


def loss():
    return tf.keras.losses.MSE(y_data, Weights*x_data+biases)


optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)


for step in range(501):
    # loss必须是个函数
    print(loss())
    optimizer.minimize(loss, var_list=[Weights, biases])
    if step % 20 == 0:
        print(f'参数W：{Weights.read_value()}, 参数B：{biases.read_value()}')

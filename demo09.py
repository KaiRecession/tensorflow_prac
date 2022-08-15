import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

SAMPLES = 1000
np.random.seed(1337)
x_values = np.random.uniform(low=0, high=2 * math.pi, size=SAMPLES)
np.random.shuffle(x_values)
y_values = np.sin(x_values)
y_values += 0.1 * np.random.randn(*y_values.shape)

TRAIN_SPLIT = int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

assert (x_train.size + x_test.size + x_validate.size) == SAMPLES
assert (y_train.size + y_test.size + y_validate.size) == SAMPLES

plt.plot(x_train, y_train, 'r.', label="Train")
plt.plot(x_test, y_test, 'g.', label="Test")
plt.plot(x_validate, y_validate, 'b.', label="Validate")

plt.legend()
plt.show()

from tensorflow import keras
#
# model = keras.Sequential()
# model.add(keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(1,)))
# model.add(keras.layers.Dense(1))
# model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
# model.summary()
# history=model.fit(x_train,y_train,epochs=1000,batch_size=16,validation_data=(x_validate,y_validate))




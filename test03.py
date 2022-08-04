from tensorflow.keras import Sequential, layers
from tensorflow.keras import losses, optimizers, datasets
import tensorflow as tf


def preprocess(x, y):
    print(x.shape, y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28, 28])
    y = tf.cast(y, dtype=tf.int64)

    return x, y


def predict(test_db1, network1):
    correct1, total1 = 0, 0
    for x1, y1 in test_db1:
        x1 = tf.expand_dims(x1, axis=3)
        out1 = network1(x1)
        pred1 = tf.argmax(out1, axis=-1)
        y1 = tf.cast(y1, tf.int64)
        correct1 += float(tf.reduce_sum(tf.cast(tf.equal(pred1, y1), tf.float32)))
        total1 += x.shape[0]
    print('test acc:', correct1 / total1)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    optimizer = optimizers.RMSprop(0.001)
    batchs = 128
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(1000).batch(batchs).map(preprocess)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.shuffle(1000).batch(batchs).map(preprocess)

    network = Sequential([
        layers.Conv2D(6, kernel_size=3, strides=1),
        layers.MaxPooling2D(pool_size=2, strides=3),
        layers.ReLU(),
        layers.Conv2D(16, kernel_size=3, strides=1),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10)
    ])
    network.build(input_shape=(4, 28, 28, 1))
    network.summary()
    criteon = losses.CategoricalCrossentropy(from_logits=True)

    for epoch in range(30):
        correct, total, loss = 0, 0, 0
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                x = tf.expand_dims(x, axis=3)
                out = network(x)
                pred = tf.argmax(out, axis=-1)
                y_onehot = tf.one_hot(y, depth=10)
                loss1 = criteon(y_onehot, out)
                print(loss1.numpy)
                loss += loss1
                correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
                total += x.shape[0]
            a = network.trainable_variables
            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))
        print(epoch, 'loss=', float(loss), 'acc=', correct / total)
    predict(test_db, network)

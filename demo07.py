import tensorflow as tf
from tensorflow.keras import layers, datasets

batches = 128


class CategoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name="my_Metric"):
        super().__init__(name=name)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")
        # 也可以调整shape一次计算多个指标
        self.b = self.add_weight(name="b", initializer="random_uniform")
        print(self.b)

    # 通过y_true 与 y_pred 实现指标更新
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred)
        values = tf.cast(tf.equal(y_pred, y_true), tf.float32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            print(sample_weight)
            values = tf.multiply(values, sample_weight)
        print(11111111)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)


class NetWork(tf.keras.Model):
    def __init__(self):
        # 继承父类的所有方法
        super().__init__()
        # 没有参数的层可以共用，model汇总参数信息是按照下面的顺序，而不是按照call函数里面的顺序
        # 用到的所有层都要在这里写好
        self.ReLu = layers.ReLU()
        self.Flatten = layers.Flatten()
        self.conv1 = layers.Conv2D(6, kernel_size=3, strides=1)
        self.pool1 = layers.MaxPooling2D(pool_size=2, strides=3)
        self.conv2 = layers.Conv2D(16, kernel_size=3, strides=1)
        self.pool2 = layers.MaxPooling2D(pool_size=2, strides=2)
        self.Dense1 = layers.Dense(120, activation='relu')
        self.Dense2 = layers.Dense(84, activation='relu')
        self.Dense3 = layers.Dense(10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.ReLu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.ReLu(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.Dense2(x)
        x = self.Dense3(x)

        return x


def losses(y_true, y_pred):
    # axis指将要扩充的纬度
    y_onehot = tf.one_hot(y_true, depth=10, axis=2)
    # 多出来一个1的纬度
    y_onehot = tf.squeeze(y_onehot)
    # print(y_pred)
    # print(y_onehot)
    loss1 = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = loss1(y_onehot, y_pred)
    return loss


def buildModel():
    model = NetWork()
    model.build(input_shape=(4, 28, 28, 1))
    model.summary()
    return model


def preprocess(x, y):
    print(x.shape, y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28, 28, 1])
    y = tf.cast(y, dtype=tf.int64)

    return x, y


def main():
    network = buildModel()
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(1000).batch(batches).map(preprocess)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.shuffle(1000).batch(batches).map(preprocess)

    network.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss=losses, metrics=[CategoricalTruePositives()])
    history = network.fit(train_db, epochs=30, validation_data=test_db, validation_freq=10)
    history.history()


if __name__ == '__main__':
    main()

import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from sklearn import model_selection
# 导入Model后就会有类的提示
# 其实导入的作用很大部分就是为了提示
from tensorflow.keras import Model


batches = 128
random.seed(128)

# 不同的网络初始化参数影响结果，seed为1的时候结果较差，而seed为128很快收敛
# 归一化最重要，不归一化激活函数很难起作用
# 数据集的范围会影响是否收敛， x如果是0-6.18就很快收敛，0-100就不好收敛
# 函数的拟合的时候，有负数，适当的中间隐藏层使用relu获得好的收敛效果
# 负数不好调理就想办法变成正数
# x的范围变大，那么归一化后，在一个周期内，函数就变得更复杂，网络非常难拟合，增加网络层数可以提高拟合能力
# 单纯的增加网络层数效果和扩大单层网络的神经元数都可以提高模型的拟合能力，一味的提高两者中的某一项可能不会效果提高很多，两者同时改变会有不一样的效果
# 可能是多层的网络层数需要单层有多个神经元，这样表达能力才能够相辅相成
# 在构建神经网络时，构建方式要与训练时的构建方式相同，即：使用tf.keras.models.Sequential([])方式的话，要都是用该方式；如果使用类进行构建的话，两个都要使用相同的类进行构建
# 网络参数的初始化、数据集batch的shuffle方式只会影响收敛速度，不会影响最终的收敛点，loss等高线的起点不同，路径不同罢了，终点是一样的。影响终点的只有网络结构了
tf.random.set_seed(3)


class NetWork(Model):
    def __init__(self):
        super(NetWork, self).__init__()
        self.Dense1 = tf.keras.layers.Dense(200, activation='relu')
        self.Dense2 = tf.keras.layers.Dense(200, activation='relu')
        self.Dense3 = tf.keras.layers.Dense(200, activation='relu')
        self.Dense4 = tf.keras.layers.Dense(200, activation='relu')
        self.Dense5 = tf.keras.layers.Dense(140, activation='relu')
        self.Dense6 = tf.keras.layers.Dense(120, activation='relu')
        self.Dense7 = tf.keras.layers.Dense(80, activation='relu')
        self.Dense8 = tf.keras.layers.Dense(1, activation='relu')

    def call(self, inputs, training=None, mask=None):
        x = self.Dense1(inputs)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)
        x = self.Dense5(x)
        x = self.Dense6(x)
        x = self.Dense7(x)
        x = self.Dense8(x)
        return x


def prepareData():
    x = []
    y = []
    for i in range(10000):
        data = random.uniform(0, 100)
        x.append(data)
        y.append(math.cos(data))
        # if math.cos(data) < 0:
        #     print(math.cos(data))
    x = np.array(x) / 100
    y = np.array(y) + 1
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=1234)
    # plt.scatter(x_train, y_train, s=1, c='red', alpha=0.5)
    # plt.show()
    return (x_train, y_train), (x_test, y_test)


def prepareDataSet():
    (x_train, y_train), (x_test, y_test) = prepareData()
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # shuffle里面的参数简单理解为混乱程度
    train_db = train_db.shuffle(1000).batch(batches)
    test_db = test_db.shuffle(1000).batch(batches)
    return train_db, test_db


def losses(y_true, y_pred):
    subLoss = tf.keras.losses.MeanSquaredError()
    loss = subLoss(y_true, y_pred)
    return loss


def train(train_db, test_db, network):
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(2000):
        correct, total, loss = 0, 0, 0
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = network(x)
                # for i in out:
                #     if i < 0:
                #         print(i)
                loss1 = losses(y, out)
                loss += loss1
            a = network.trainable_variables
            grads = tape.gradient(loss1, a)
            optimizer.apply_gradients(zip(grads, a))
        if (epoch + 1) % 10 == 0:
            network.save_weights('./model/cos2.ckpt')
            print('model is save')
        print(epoch, 'loss =', float(loss))


def main():
    train_db, test_db = prepareDataSet()
    network = NetWork()
    network.build(input_shape=(10, 1))
    network.summary()
    network.load_weights('./model/cos2.ckpt')
    train(train_db, test_db, network)
    print("end")


def valid():
    network = NetWork()
    network.load_weights('./model/cos2.ckpt')
    x = []
    x = np.arange(0, 100, 0.0001)
    x = tf.convert_to_tensor(x)
    x = tf.reshape(x, (-1, 1))
    y = network((x / 100))
    plt.scatter(x, y, s=1, c='red', alpha=0.5)
    plt.show()

    # print(y)


def convertModel():
    import coremltools as ct
    kerasModel = NetWork()
    kerasModel.build(input_shape=(1, 1))
    kerasModel.compute_output_shape(input_shape=(1, 1))
    kerasModel.load_weights('./model/cos2.ckpt')
    print(kerasModel.predict([[0.0]]))
    input = ct.TensorType(shape=(1, 1))
    converter = ct.convert(kerasModel, inputs=[input])
    converter.save('./coreml/cos.mlpackage')


def saveModel():
    kerasModel = NetWork()
    kerasModel.build(input_shape=(1, 1))
    kerasModel.compute_output_shape(input_shape=(1, 1))
    kerasModel.load_weights('./model/cos2.ckpt')
    tf.saved_model.save(kerasModel, './savedModel')
    # model = tf.saved_model.load(path_to_dir)


def jsModel():
    model = tf.saved_model.load('./savedModel')
    print(np.reshape([float(1.1)], (1, 1)))
    print(model(tf.reshape([float(1.1)], (1, 1))))


if __name__ == '__main__':
    print(tf.__version__)
    # main()
    # valid()
    # saveModel()
    jsModel()

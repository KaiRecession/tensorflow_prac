import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_physical_devices('GPU')
            print(len(gpus), "Physical GPUS,", len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)
    model = keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ])
    # 第一层参数的计算方法：784 * 256 + 256，4就带遍一个batch的size？？？，应该是吧
    model.build(input_shape=(4, 784))
    model.summary()

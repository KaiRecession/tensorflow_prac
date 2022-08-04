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
    # 单卷积核多通道也会翻倍参数，多卷积核又会翻倍参数，但是一个偏置参数配一个卷积核
    # padding是填充，在原来的输入上填充0，只会在strides为0的时候保证输入输入大小一样
    # 输出：batch，长，宽，卷积核数（多通道在单卷积核上会相加的）
    model = keras.Sequential([
        layers.Conv2D(4, kernel_size=3, strides=1, padding='SAME')
    ])
    # 第一层参数的计算方法：784 * 256 + 256，4就带遍一个batch的size？？？，应该是吧
    model.build(input_shape=(4, 10, 10, 3))
    model.summary()

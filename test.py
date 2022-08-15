import tensorflow as tf


y_true = [[0, 1, 0], [0, 1, 0]]
y_pred = [[0.05, 0.95, 0], [0.05, 0.95, 0]]

y = [2, 3, 4]
y = tf.one_hot(y, depth=5, axis=1)
# Using 'auto'/'sum_over_batch_size' reduction type.
cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
# 内置了求平均或者求和，默认是求平均
cce = tf.keras.losses.CategoricalCrossentropy()
# 真实标签放在前面，真实标签分别乘上loge（预测元素），负数求和就是交叉墒
print(cce(y_true, y_pred).numpy())

# 注意b的维度：(3,)
b = tf.constant([1, 2, 3])
# 创建标量的方法
a = tf.zeros([])
print(a)
print(b)
print(f'a的值：{a},a的维度：{a.shape}')
print(f'b的值：{b},b的维度：{b.shape}')
print(a.shape)
print(b.shape)

a = tf.constant([[2, 3, 4],
     [3, 2, 4]])
b = tf.constant([10, 20, 30])
c = a + b
print(c)

x = tf.linspace(-8., 8, 20)
y = tf.linspace(8., 16, 100)
x, y = tf.meshgrid(x, y)
print(x, y)

import tensorflow as tf

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

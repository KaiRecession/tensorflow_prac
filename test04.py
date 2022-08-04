import tensorflow as tf
# 直接对应的相乘
y_true = [[0, 1, 0], [0, 1, 0]]
y_pred = [[0.03, 0.95, 0.02], [0.10, 0.85, 0.05]]
# Using 'auto'/'sum_over_batch_size' reduction type.
# 算的是平均损失
cce = tf.keras.losses.CategoricalCrossentropy()
a = cce(y_true, y_pred).numpy()
print(a)
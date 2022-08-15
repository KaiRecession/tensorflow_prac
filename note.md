## Pycharm提示问题

找到conda的python文件夹，再找到site-packages\tensorflow

MAC的路径是：/Users/kevin/miniforge3/lib/python3.9/site-packages/tensorflow/__init__.py

在最后添加以下代码：

```python
from tensorflow import contrib as contrib
from tensorflow.python.util.lazy_loader import LazyLoader  
# pylint: disable=g-import-not-at-top
contrib = LazyLoader('contrib', globals(), 'tensorflow.contrib')
del LazyLoader
```

报错了就再删除，反正提示就有了

提示前在前面把类的导入语句写了

## 全连接层

![截屏2022-08-08 16.21.04](/Users/kevin/Documents/MyCode/tensorflow_practice/demo01/img/截屏2022-08-08 16.21.04.png)

每个神经元都连接上一层的所有神经元，所以参数数量很大，激活函数只针对每一个神经元的求和，激活函数就和偏置数量是一样的。激活函数是在求和之后才进行计算。

神经元里的参数个数 = input个数

![截屏2022-08-08 16.43.22](/Users/kevin/Documents/MyCode/tensorflow_practice/demo01/img/截屏2022-08-08 16.43.22.png)

![截屏2022-08-08 16.29.51](/Users/kevin/Documents/MyCode/tensorflow_practice/demo01/img/截屏2022-08-08 16.29.51.png)

隐藏层就是除去输入和输出

### SoftMax激活函数

![截屏2022-08-08 16.48.02](/Users/kevin/Documents/MyCode/tensorflow_practice/demo01/img/截屏2022-08-08 16.48.02.png)

## 卷积层

全连接层每一个神经元会为每一个输入设定一个参数，卷积层的卷积核就相当于全连接层的一个神经元，卷积层的神经元的参数取决于kennelSize，每多一个通道多**一个**神经元，每多一个卷积核就多**一倍**的神经元

神经元里的参数 = kennelSize

输出看是否padding

单通道、单卷积核输出

![截屏2022-08-08 17.19.32](/Users/kevin/Documents/MyCode/tensorflow_practice/demo01/img/截屏2022-08-08 17.19.32.png)

多通道、单卷积核输出

![截屏2022-08-08 17.19.47](/Users/kevin/Documents/MyCode/tensorflow_practice/demo01/img/截屏2022-08-08 17.19.47.png)

多通道、多卷积核输出

![截屏2022-08-08 17.20.04](/Users/kevin/Documents/MyCode/tensorflow_practice/demo01/img/截屏2022-08-08 17.20.04.png)

有padding时的输出

![截屏2022-08-08 17.20.24](/Users/kevin/Documents/MyCode/tensorflow_practice/demo01/img/截屏2022-08-08 17.20.24.png)

## 池化层

池化层加在卷积层后面，用来达到缩减参数的作用

![截屏2022-08-08 17.44.36](/Users/kevin/Documents/MyCode/tensorflow_practice/demo01/img/截屏2022-08-08 17.44.36.png)

## BatchNorm层（BN层）

看损失等高线，当参数的分布特性不一样的时候，等高线就不圆，越圆的等高线就能越快进行收敛。具体的优化看李宏毅课程的笔记

![截屏2022-08-08 21.28.18](/Users/kevin/Documents/MyCode/tensorflow_practice/demo01/img/截屏2022-08-08 21.28.18.png)

## 关于SGD的想法

目的是为了寻找最优的参数，BGD就是把所有的数据输入，然后求系数的梯度方向 ，等于说旧的参数（系数）跑完了所有的数据梯度方向（相当于平均方向），按照方向更新了参数，这些梯度方向都是在**上一次的参数求的**。而SGD时参数就跑了一个Batch的梯度方向，下一次的Batch就会用更新后的系数。

## 激活函数

激活函数的作用是**增加模型的非线性表达能力**，不加激活函数就是纯线性组合了，就好比之前

想办法把标签值变为正数

```
# 不同的网络初始化参数影响结果，seed为1的时候结果较差，而seed为128很快收敛
# 归一化最重要，不归一化激活函数很难起作用
# 数据集的范围会影响是否收敛， x如果是0-6.18就很快收敛，0-100就不好收敛
# 函数的拟合的时候，有负数，适当的中间隐藏层使用relu获得好的收敛效果
# 负数不好调理就想办法变成正数
# x的范围变大，那么归一化后，在一个周期内，函数就变得更复杂，网络非常难拟合，增加网络层数可以提高拟合能力
# 单纯的增加网络层数效果和扩大单层网络的神经元数都可以提高模型的拟合能力，一味的提高两者中的某一项可能不会效果提高很多，两者同时改变会有不一样的效果
# 可能是多层的网络层数需要单层有多个神经元，这样表达能力才能够相辅相成
# 图像的x取值范围就是255个不同，曲线的复杂程度远小于cos周期压缩几十个周期
```

### 两种结构的模型

在Keras中有两种深度学习的模型：序列模型（Sequential）和通用模型（Model）。差异在于不同的拓扑结构。序列模型各层之间是依次顺序的线性关系，模型结构通过一个列表来制定。通用模型可以设计非常复杂、任意拓扑结构的神经网络，例如有向无环网络、共享层网络等。相比于序列模型只能依次线性逐层添加，通用模型能够比较灵活地构造网络结构，设定各层级的关系
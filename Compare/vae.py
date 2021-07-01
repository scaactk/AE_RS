import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

# VAE是对编码表示添加的约束
# 从高斯分布到 离散， 怎么离散？

batch_size = 100

x = layers.Input(shape=(784,))
h = layers.Dense(256, activation='relu')(x)
# 将输入映射到潜在分布的参数上
z_mean = layers.Dense(2)(h)
z_log_sigma = layers.Dense(2)(h)


def sampling(args):
    z_mean, z_log_sigma = args
    epslion = keras.backend.random_normal(shape=(batch_size, 2))
    return z_mean + keras.backend.exp(z_log_sigma) * epslion


# 如果某一层需要通过一个函数去变换数据，那利用keras.layers.Lambda()这个函数单独把这一步数据操作命为单独的一Lambda层
# 最后括号里的是这层的输入，和普通的层写法一样
z = layers.Lambda(sampling, output_shape=(2,))([z_mean, z_log_sigma])

# 这里定义层的时候，后面并没有写输入的参数，不知道为什么要单独写这两层
decoder_h = layers.Dense(256, activation='relu')
decoder_mean = layers.Dense(784, activation='sigmoid')
# 根据采样重建
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):
    print(tf.shape(x))
    xent_loss = keras.backend.sum(keras.backend.binary_crossentropy(x, x_decoded_mean), axis=-1)
    kl_loss = - 0.5 * keras.backend.sum(1 + z_log_sigma - keras.backend.square(z_mean) - keras.backend.exp(z_log_sigma),
                                        axis=-1)
    return xent_loss + kl_loss


vae = keras.Model(x, x_decoded_mean)
encoder = keras.Model(x, z)
vae.compile(optimizer=keras.optimizers.RMSprop(), loss=vae_loss, run_eagerly=False)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=50,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# 可视化
import matplotlib.pyplot as plt

# 最后的batch_size是分批预测，默认是32，比单个预测要快
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

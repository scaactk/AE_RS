from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.reshape(len(x_test), 28, 28, 1)
x_test = x_test.astype('float32') / 255.0

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# np.clip截取范围，使得小于0的变成0，大于1的变成1，其他不变，防止图像里数据变成负的
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# 查看加了噪声的数据
import matplotlib.pyplot as plt

# n = 10
# plt.figure(figsize=(20, 2))
# for i in range(n):
#     ax = plt.subplot(1, n, i + 1)
#     plt.imshow(x_test_noisy[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

input = keras.Input(shape=(28, 28, 1))

# 为了提高重建图像的质量，这次使用了更多的滤波器
x = layers.Conv2D(32, 3, activation='relu', padding='same')(input)
x = layers.MaxPool2D(2, padding='same')(x)
x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
encoded = layers.MaxPool2D(2, padding='same')(x)   # 7*7*32

x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoded)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = layers.UpSampling2D(2)(x)
decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

model = keras.Model(input, decoded)
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.binary_crossentropy,
              )

from tensorflow.keras.callbacks import TensorBoard
model.fit(x_train_noisy, x_train,
          epochs=50,
          batch_size=512,
          shuffle=True,
          validation_data=(x_test_noisy, x_test_noisy),
          callbacks=[TensorBoard(log_dir='tmp/dae', histogram_freq=0, write_graph=False)],
          )

decoded_imgs = model.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()








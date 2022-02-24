# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 00:57:23 2020

@author: Lenovo
"""

from keras.layers import Input, Dense
from keras.models import Model
import warnings
warnings.filterwarnings('ignore')
# 编码潜在空间表征维度
encoding_dim = 32 
 # 自编码器输入
input_img = Input(shape=(784,))
# 使用一个全连接网络来搭建编码器
encoded = Dense(encoding_dim, activation='relu')(input_img)
# 使用一个全连接网络来对编码器进行解码
decoded = Dense(784, activation='sigmoid')(encoded)
# 构建keras模型
autoencoder = Model(input=input_img, output=decoded)

# 编码器模型
encoder = Model(input=input_img, output=encoded)
# 解码器模型
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

# 编译模型
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# 准备mnist数据
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))# 训练
autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

import matplotlib.pyplot as plt

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
n = 10  
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # 展示原始图像
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 展示自编码器重构后的图像
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

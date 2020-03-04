from __future__ import absolute_import

import tensorflow as tf

from tf_pose import network_base

class ResBlock(tf.keras.Model):

    def __init__(self,filter_num,stride=1):
        super(ResBlock, self).__init(name="ResBlock")
        self.conv1=tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3,3), strides=stride,padding="same")
        self.bn1=tf.keras.layers.BatchNormalziation()
        self.conv2=tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3,3), strides=1, padding="same")
        self.bn2=tf.keras.layers.BatchNormalziation()
        ##relu activation
        self.activation=tf.keras.layers.ReLU()
        self.stride=stride
        ##dot-line connection by using 1*1 kernel size for projection
        if self.stride!=1:
            self.conv3=tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1,1), strides, padding="valid")
            self.bn3=tf.keras.layers.BatchNormalziation()
    def call(self,x):
        x1=self.conv1(x)
        x1=self.bn1(x1)
        x1=self.activation(x1)
        x1=self.conv2(x1)
        x1=self.bn2(x1)
        ##match the increasing dimensions
        if self.stride!=1:
            x=self.conv3(x)
            x=self.bn3(x)
        x1=tf.keras.layers.add([x,x1])
        x1=self.activation(x1)
        return x1

class ResNetNetwork(tf.keras.Model):
    def __init__(self):
        super(ResNetwork, self).__init__(name='ResNet34')
        self.conv1=tf.keras.layers(filters=64, kernel_size=(7,7),strides=2, padding="same")
        self.bn1=BatchNormalziation()
        self.activation=tf.keras.layers.ReLU()
        self.maxpool=tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding="valid")

        #based on the resnet paper, 3 blocks of 64 feature maps
        self.conv2_1=ResBlock(64)
        self.conv2_2=ResBlock(64)
        self.conv2_3=ResBlock(64)

        #4 blocks of 128 feature maps
        self.conv3_1=ResBlock(128,2)
        self.conv3_2=ResBlock(128)
        self.conv3_3=ResBlock(128)
        self.conv3_4=ResBlock(128)

        #6 blocks of 128 feature maps
        self.conv4_1=ResBlock(256,2)
        self.conv4_2=ResBlock(256)
        self.conv4_3=ResBlock(256)
        self.conv4_4=ResBlock(256)
        self.conv4_5=ResBlock(256)
        self.conv4_6=ResBlock(256)

        #based on the resnet paper, 3 blocks of 64 feature maps
        self.conv5_1=ResBlock(512,2)
        self.conv5_2=ResBlock(512)
        self.conv5_3=ResBlock(512)
        #omit the final pooling and fc layers due to our requirements
        #self.apool=tf.keras.layers.GlobalAvgPool2D()
        #self.fc=...
        #dimensions reduction
        #self.maxpool2=tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding="valid")
        #self.maxpool2=tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding="valid")
    def call(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        return x

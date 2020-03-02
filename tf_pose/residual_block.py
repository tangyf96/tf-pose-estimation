import tensorflow as tf

class ResBlock(tf.keras.layers.Layer):

    def __init__(self,filter_num,stride=1):
        super(ResBlock, self).__init()
        self.conv1=tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3,3), strides=stride,padding="same")
        self.bn1=tf.keras.layers.BatchNormalziation()
        self.conv2=tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3,3), strides=1, padding="same")
        self.bn2=tf.keras.layers.BatchNormalziation()

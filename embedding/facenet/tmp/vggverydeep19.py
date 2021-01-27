"""Load the VGG imagenet model into TensorFlow.
Download the model from http://www.robots.ox.ac.uk/~vgg/research/very_deep/
and point to the file 'imagenet-vgg-verydeep-19.mat'
"""
import numpy as np
from scipy import io
import tensorflow as tf

def load(filename, images):
    vgg19 = io.loadmat(filename)
    vgg19Layers = vgg19['layers']
    
    # A function to get the weights of the VGG layers
    def vbbWeights(layerNumber):
        W = vgg19Layers[0][layerNumber][0][0][2][0][0]
        W = tf.constant(W)
        return W
      
    def vbbConstants(layerNumber):
        b = vgg19Layers[0][layerNumber][0][0][2][0][1].T
        b = tf.constant(np.reshape(b, (b.size)))
        return b
    
    modelGraph = {}
    modelGraph['input'] = images
    modelGraph['conv1_1'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['input'], filters = vbbWeights(0), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(0))
    modelGraph['conv1_2'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['conv1_1'], filters = vbbWeights(2), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(2))
    modelGraph['avgpool1'] = tf.nn.avg_pool2d(input=modelGraph['conv1_2'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    modelGraph['conv2_1'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['avgpool1'], filters = vbbWeights(5), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(5))
    modelGraph['conv2_2'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['conv2_1'], filters = vbbWeights(7), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(7))
    modelGraph['avgpool2'] = tf.nn.avg_pool2d(input=modelGraph['conv2_2'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    modelGraph['conv3_1'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['avgpool2'], filters = vbbWeights(10), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(10))
    modelGraph['conv3_2'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['conv3_1'], filters = vbbWeights(12), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(12))
    modelGraph['conv3_3'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['conv3_2'], filters = vbbWeights(14), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(14))
    modelGraph['conv3_4'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['conv3_3'], filters = vbbWeights(16), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(16))
    modelGraph['avgpool3'] = tf.nn.avg_pool2d(input=modelGraph['conv3_4'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    modelGraph['conv4_1'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['avgpool3'], filters = vbbWeights(19), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(19))
    modelGraph['conv4_2'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['conv4_1'], filters = vbbWeights(21), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(21))
    modelGraph['conv4_3'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['conv4_2'], filters = vbbWeights(23), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(23))
    modelGraph['conv4_4'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['conv4_3'], filters = vbbWeights(25), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(25))
    modelGraph['avgpool4'] = tf.nn.avg_pool2d(input=modelGraph['conv4_4'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    modelGraph['conv5_1'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['avgpool4'], filters = vbbWeights(28), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(28))
    modelGraph['conv5_2'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['conv5_1'], filters = vbbWeights(30), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(30))
    modelGraph['conv5_3'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['conv5_2'], filters = vbbWeights(32), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(32))
    modelGraph['conv5_4'] = tf.nn.relu(tf.nn.conv2d(input=modelGraph['conv5_3'], filters = vbbWeights(34), strides = [1, 1, 1, 1], padding = 'SAME') + vbbConstants(34))
    modelGraph['avgpool5'] = tf.nn.avg_pool2d(input=modelGraph['conv5_4'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    return modelGraph


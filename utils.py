import tensorflow as tf
import numpy as np

class avg_pool_res(object):
	def __init__(self, width, stride):
		self.width = width
		self.stride = stride

	def activated_on(self, x):
		x = tf.expand_dims(x, [1])
		avg_pool = tf.nn.avg_pool(x, ksize=[1, 1, self.width, 1], strides = [1, 1, self.stride, 1], padding="SAME")
		return tf.squeeze(avg_pool, [1])

class Conv_1x1(object):
	def __init__(self, var_scope, in_channels, out_channels, init_bias = 0.0):
		self.var_scope = var_scope
		self.in_channels = in_channels
		self.out_channels = out_channels

		with tf.variable_scope(self.var_scope):
			with tf.variable_scope("conv_1x1"):
				self.filter = tf.get_variable("filter", [1, in_channels, out_channels], tf.float32, tf.contrib.layers.variance_scaling_initializer())
				self.bias = tf.get_variable("bias", [1, out_channels], tf.float32, tf.constant_initializer(init_bias))

	def activated_on(self, x):
		with tf.variable_scope(self.var_scope):
			with tf.variable_scope("conv_1x1"):
				conv_act = tf.nn.conv1d(x, self.filter, stride = 1, padding = "SAME")
				return conv_act + self.bias

class Causal_Conv(object):
	def __init__(self, var_scope, filter_width, in_channels, out_channels, dilation):
		self.var_scope = var_scope
		self.filter_width = filter_width
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.dilation = dilation

		with tf.variable_scope(self.var_scope):
			self.filter = tf.get_variable("filter", [self.filter_width, self.in_channels, self.out_channels], tf.float32, tf.contrib.layers.variance_scaling_initializer())
			self.bias = tf.get_variable("bias", [1, self.out_channels], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x):
		with tf.variable_scope(self.var_scope):
			padding = [[0, 0], [(self.filter_width - 1) * self.dilation, 0], [0, 0]]
			padded = tf.pad(x, padding)
			conv_act = tf.nn.convolution(padded, self.filter, padding="VALID", dilation_rate = [self.dilation])
			return conv_act + self.bias

class Dense(object):
	def __init__(self, name, in_dim, out_dim):
		with tf.variable_scope(name):
			self.W = tf.get_variable("W", [in_dim, out_dim], tf.float32, tf.contrib.layers.variance_scaling_initializer())
			self.b = tf.get_variable("b", [out_dim], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x):
		return tf.matmul(x, self.W) + self.b

EPSILON = 1e-7
TWO_PI = 3.141592653 * 2

def log_p_norm(obs, mu_param, var_param):
	return (-0.5 * tf.log(TWO_PI * (var_param + EPSILON))) - (tf.square(obs - mu_param) / ((2 * var_param) + EPSILON))

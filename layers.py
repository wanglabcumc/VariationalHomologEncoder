import tensorflow as tf
import numpy as np

class Conv(object):
	#Vanilla convolution, no batch-normalization.
	def __init__(self, var_scope, filter_width, in_channels, out_channels, dilation = 1):
		self.var_scope = var_scope
		self.filter_width = filter_width
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.dilation = dilation

		with tf.variable_scope(self.var_scope):
			self.filter = tf.get_variable("filter", [self.filter_width, self.in_channels, self.out_channels], tf.float32, tf.random_normal_initializer(0.0, 0.05)) #tf.contrib.layers.variance_scaling_initializer())
			self.bias = tf.get_variable("bias", [1, self.out_channels], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x, strides=None):
		with tf.variable_scope(self.var_scope):
			if strides is None:
				conv_act = tf.nn.convolution(input = x, filter = self.filter, padding="SAME", dilation_rate = [self.dilation])
			else:
				conv_act = tf.nn.convolution(input = x, filter = self.filter, padding="SAME", strides=strides, dilation_rate = [self.dilation])
			return conv_act + self.bias

class Conv_1x1(object):
	def __init__(self, var_scope, in_channels, out_channels):
		self.var_scope = var_scope
		self.in_channels = in_channels
		self.out_channels = out_channels

		with tf.variable_scope(self.var_scope):
			with tf.variable_scope("conv_1x1"):
				self.filter = tf.get_variable("filter", [1, in_channels, out_channels], tf.float32, tf.random_normal_initializer(0.0, 0.05)) #tf.contrib.layers.variance_scaling_initializer())
				self.bias = tf.get_variable("bias", [1, out_channels], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x):
		with tf.variable_scope(self.var_scope):
			with tf.variable_scope("conv_1x1"):
				conv_act = tf.nn.conv1d(x, self.filter, stride = 1, padding = "SAME")
				return conv_act + self.bias

class Deconv_1D(object):
	def __init__(self, prefix, in_width, out_width, in_channels, out_channels, stride, batch_size):
		self.batch_size = batch_size
		self.in_width = in_width
		self.in_channels = in_channels
		self.out_width = out_width
		self.out_channels = out_channels
		self.stride = stride
		self.filter = tf.get_variable("%s_deconv_filter" %(prefix), [1, 4, out_channels, in_channels], tf.float32, tf.random_normal_initializer(0.0, 0.05)) #tf.contrib.layers.variance_scaling_initializer())
		self.bias = tf.get_variable("%s_bias" %(prefix), [out_channels], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x):
		x = tf.expand_dims(x, [1]) #reshape(x, [self.batch_size, 1, self.in_width, self.in_channels])
		pre_act_x = tf.nn.conv2d_transpose(x, self.filter, output_shape = [self.batch_size, 1, self.out_width, self.out_channels], strides=self.stride)
		pre_act_x = tf.squeeze(pre_act_x, [1])
		return pre_act_x + self.bias

class Dense(object):
	def __init__(self, var_scope, in_width, out_width):
		self.var_scope = var_scope
		self.in_width = in_width
		self.out_width = out_width

		with tf.variable_scope(self.var_scope):
			self.W = tf.get_variable("dense_W", [self.in_width, self.out_width], tf.float32, tf.random_normal_initializer(0.0, 0.05)) #tf.contrib.layers.variance_scaling_initializer())
			self.beta = tf.get_variable("dense_beta", [1, self.out_width], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x, init = False):
		with tf.variable_scope(self.var_scope):
			pre_act = tf.matmul(x, self.W)
			return pre_act + self.beta

class MLP(object):
	def __init__(self, name, in_dim, out_dim):
		self.var_scope = name
		with tf.variable_scope(self.var_scope):
			self.i_to_h = Dense("mlp_hid", in_dim, out_dim)
			self.h_to_out = Dense("mlp_out", out_dim, out_dim)

	def activated_on(self, x):
		hid = tf.nn.relu(self.i_to_h.activated_on(x))
		out = self.h_to_out.activated_on(hid)
		return out

#Two layer MLP with two outputs for some NN needs.
class MLP2(object):
	def __init__(self, name, in_dim, out_dim):
		self.var_scope = name
		with tf.variable_scope(self.var_scope):
			self.i_to_h = Dense("mlp2_hid", in_dim, out_dim)
			self.h_to_alpha = Dense("mlp2_alpha", out_dim, out_dim)
			self.h_to_beta = Dense("mlp2_beta", out_dim, out_dim)

	def activated_on(self, x):
		hid = tf.nn.relu(self.i_to_h.activated_on(x))
		alpha = self.h_to_alpha.activated_on(hid)
		beta = self.h_to_beta.activated_on(hid)
		return alpha, beta

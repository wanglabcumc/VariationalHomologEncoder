import tensorflow as tf
import numpy as np

EPSILON = 1e-6

class BN_Conv(object):
	def __init__(self, var_scope, is_training, filter_width, in_channels, out_channels, dilation = 1):
		self.var_scope = var_scope
		self.filter_width = filter_width
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.dilation = dilation
		self.is_training = is_training
		self.decay = 0.9999

		with tf.variable_scope(self.var_scope):
			self.batch_mean_ema = tf.get_variable("mean_ema", [1, self.out_channels], tf.float32, tf.constant_initializer(0.0), trainable = False)
			self.batch_var_ema = tf.get_variable("var_ema", [1, self.out_channels], tf.float32, tf.constant_initializer(1.0), trainable = False) #we need to implement this for validation/test.

			self.filters = tf.get_variable("bn_filter", [self.filter_width, self.in_channels, self.out_channels], tf.float32, tf.random_normal_initializer(0.0, 0.05)) #tf.contrib.layers.variance_scaling_initializer())
			self.gamma = tf.get_variable("bn_gamma", [1, self.out_channels], tf.float32, tf.constant_initializer(1.0))
			self.beta = tf.get_variable("bn_beta", [1, self.out_channels], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x, strides = [1], padding = "SAME"):
		with tf.variable_scope(self.var_scope):
			pre_act = tf.nn.convolution(input = x, filter = self.filters, padding = padding, strides = strides, dilation_rate = [self.dilation])
			if self.is_training:
				batch_mean, batch_var = tf.nn.moments(pre_act, [0, 1])

				update_mean = tf.assign(self.batch_mean_ema, self.batch_mean_ema * self.decay + batch_mean * (1. - self.decay))
				update_var = tf.assign(self.batch_var_ema, self.batch_var_ema * self.decay + batch_var * (1. - self.decay))

				with tf.control_dependencies([update_mean, update_var]):
					transformed = (pre_act - batch_mean) / tf.sqrt(batch_var + EPSILON)
					out = self.gamma * transformed + self.beta
					return out

			else:
				transformed = (pre_act - self.batch_mean_ema) / tf.sqrt(self.batch_var_ema + EPSILON)
				out = self.gamma * transformed + self.beta
				return out

class BN_Conv_1x1(object):
	def __init__(self, var_scope, is_training, in_channels, out_channels):
		self.var_scope = var_scope
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.is_training = is_training
		self.decay = 0.9999

		with tf.variable_scope(self.var_scope):
			self.batch_mean_ema = tf.get_variable("mean_ema", [1, self.out_channels], tf.float32, tf.constant_initializer(0.0), trainable = False)
			self.batch_var_ema = tf.get_variable("var_ema", [1, self.out_channels], tf.float32, tf.constant_initializer(1.0), trainable = False) #we need to implement this for validation/test.

			self.W = tf.get_variable("bn_1x1_filter", [1, self.in_channels, self.out_channels], tf.float32, tf.random_normal_initializer(0.0, 0.05)) #tf.contrib.layers.variance_scaling_initializer())
			self.gamma = tf.get_variable("bn_1x1_gamma", [1, self.out_channels], tf.float32, tf.constant_initializer(1.0))
			self.beta = tf.get_variable("bn_1x1_beta", [1, self.out_channels], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x, stride = 1, padding = "SAME"):
		with tf.variable_scope(self.var_scope):
			pre_act_x = tf.nn.conv1d(x, self.W, stride = stride, padding = padding)
			if self.is_training:
				batch_mean, batch_var = tf.nn.moments(pre_act_x, [0, 1])

				update_mean = tf.assign(self.batch_mean_ema, self.batch_mean_ema * self.decay + batch_mean * (1. - self.decay))
				update_var = tf.assign(self.batch_var_ema, self.batch_var_ema * self.decay + batch_var * (1. - self.decay))

				with tf.control_dependencies([update_mean, update_var]):
					transformed = (pre_act_x - batch_mean) / tf.sqrt(batch_var + EPSILON)
					out = self.gamma * transformed + self.beta
					return out

			else: #then we're in validation/test and we want to use
				transformed = (pre_act_x - self.batch_mean_ema) / tf.sqrt(self.batch_var_ema + EPSILON)
				out = self.gamma * transformed + self.beta
				return out

class BN_Deconv_1D(object):
	def __init__(self, prefix, is_training, in_width, out_width, in_channels, out_channels, stride, batch_size):
		self.batch_size = batch_size
		self.in_width = in_width
		self.in_channels = in_channels
		self.out_width = out_width
		self.out_channels = out_channels
		self.stride = stride
		self.var_scope = "deconv_%s"
		self.is_training = is_training
		self.decay = 0.9999

		with tf.variable_scope(self.var_scope):
			self.batch_mean_ema = tf.get_variable("mean_ema", [self.out_channels], tf.float32, tf.constant_initializer(0.0), trainable = False)
			self.batch_var_ema = tf.get_variable("var_ema", [self.out_channels], tf.float32, tf.constant_initializer(1.0), trainable = False) #we need to implement this for validation/test.

			self.filter = tf.get_variable("deconv_filter", [1, out_width // in_width, out_channels, in_channels], tf.float32, tf.random_normal_initializer(0.0, 0.05)) #tf.contrib.layers.variance_scaling_initializer())
			self.gamma = tf.get_variable("gamma", [out_channels], tf.float32, tf.constant_initializer(1.0))
			self.beta = tf.get_variable("beta", [out_channels], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x, r_max = None, d_max = None):
		#x = tf.reshape(x, [self.batch_size, 1, self.in_width, self.in_channels])
		with tf.variable_scope(self.var_scope):
			x = tf.expand_dims(x, [1]) #adds a height dimension.
			pre_act = tf.nn.conv2d_transpose(x, self.filter, output_shape = [self.batch_size, 1, self.out_width, self.out_channels], strides=self.stride)
			pre_act = tf.squeeze(pre_act, [1])

			if self.is_training:
				batch_mean, batch_var = tf.nn.moments(pre_act, [0, 1])

				update_mean = tf.assign(self.batch_mean_ema, self.batch_mean_ema * self.decay + batch_mean * (1. - self.decay))
				update_var = tf.assign(self.batch_var_ema, self.batch_var_ema * self.decay + batch_var * (1. - self.decay))

				with tf.control_dependencies([update_mean, update_var]):
					transformed = (pre_act - batch_mean) / tf.sqrt(batch_var + EPSILON)
					out = self.gamma * transformed + self.beta
					return out

			else:
				transformed = (pre_act_x - self.batch_mean_ema) / tf.sqrt(self.batch_var_ema + EPSILON)
				out = self.gamma * transformed + self.beta
				return out

class BN_Dense(object):
	"""Batch_Norm_Dense :: inherits object
	Simple dense matrix multiplication layer with batch normalization."""

	def __init__(self, var_scope, is_training, in_width, out_width):
		"""__init__(self, var_scope, in_width, out_width):
		var_scope -> names for variables. Good for restoration of variables after training/organization.
		in_width/out_width -> width of input/hidden layers for the dense operation."""
		self.var_scope = var_scope
		self.in_width = in_width
		self.out_width = out_width
		self.is_training = is_training
		self.decay = 0.9999

		with tf.variable_scope(self.var_scope):
			self.batch_mean_ema = tf.get_variable("mean_ema", [1, self.out_width], tf.float32, tf.constant_initializer(0.0), trainable = False)
			self.batch_var_ema = tf.get_variable("var_ema", [1, self.out_width], tf.float32, tf.constant_initializer(1.0), trainable = False) #we need to implement this for validation/test.

			self.W = tf.get_variable("bn_dense_W", [self.in_width, self.out_width], tf.float32, tf.random_normal_initializer(0.0, 0.05)) #tf.contrib.layers.variance_scaling_initializer())
			self.gamma = tf.get_variable("bn_dense_gamma", [1, self.out_width], tf.float32, tf.constant_initializer(1.0))
			self.beta = tf.get_variable("bn_dense_beta", [1, self.out_width], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x, init=False):
		with tf.variable_scope(self.var_scope):
			pre_act = tf.matmul(x, self.W)
			if self.is_training:
				batch_mean, batch_var = tf.nn.moments(pre_act, [0])

				update_mean = tf.assign(self.batch_mean_ema, self.batch_mean_ema * self.decay + batch_mean * (1. - self.decay))
				update_var = tf.assign(self.batch_var_ema, self.batch_var_ema * self.decay + batch_var * (1. - self.decay))

				with tf.control_dependencies([update_mean, update_var]):
					transformed = (pre_act - batch_mean) / tf.sqrt(batch_var + EPSILON)
					normalized = self.gamma * transformed + self.beta
					return normalized

			else:
				transformed = (pre_act - self.batch_mean_ema) / tf.sqrt(self.batch_var_ema + EPSILON)
				return self.gamma * transformed + self.beta

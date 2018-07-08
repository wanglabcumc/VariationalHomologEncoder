import tensorflow as tf
import numpy as np

from layers import *

class LN_Conv_1x1(object):
	def __init__(self, name, in_dim, out_dim):
		self.var_scope = name
		self.in_channels = in_dim
		self.out_channels = out_dim

		with tf.variable_scope(self.var_scope):
			with tf.variable_scope("LN_conv_1x1"):
				self.filter = tf.get_variable("filter", [1, self.in_channels, self.out_channels], tf.float32, tf.contrib.layers.variance_scaling_initializer())
				self.gamma = tf.get_variable("gamma", [1, self.out_channels], tf.float32, tf.constant_initializer(1.0))
				self.beta = tf.get_variable("beta", [1, self.out_channels], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x):
		with tf.variable_scope(self.var_scope):
			with tf.variable_scope("LN_conv_1x1"):
				conv_act = tf.nn.conv1d(x, self.filter, stride = 1, padding = "SAME")
				mu, var = tf.nn.moments(conv_act, [1], keep_dims = True)
				normed = (conv_act - mu) / tf.sqrt(var + 1e-8)
				return self.gamma * normed + self.beta

#I think this is my version of the idea in paper... just a 1x1 conv with a reshape.
class Learned_Upsample(object):
	def __init__(self, scope, input_dim, up_factor):
		self.up_factor = up_factor
		self.var_scope = "learn_perf_%s" %(scope)

		with tf.variable_scope(self.var_scope): #Don't use layer norm it's not good.
			self.up_sampler = Conv_1x1("perf_up", input_dim, input_dim * up_factor)

	def activated_on(self, x):
		batch_size, width, channels = x.get_shape().as_list()
		first_act = self.up_sampler.activated_on(x)
		reshaped = tf.reshape(first_act, [batch_size, width * self.up_factor, channels])
		return reshaped #think it's just that simple.

#I think this is equivalent but presumably faster as it's more direct.
class Learned_Deconv(object):
	def __init__(self, scope, input_dim, up_factor):
		self.up_factor = up_factor
		self.var_scope = "learn_deconv_%s" %(scope)

		with tf.variable_scope(self.var_scope):
			self.filter = tf.get_variable("deconv_up", [1, up_factor, input_dim, input_dim], tf.float32, tf.contrib.layers.variance_scaling_initializer())
			self.bias = tf.get_variable("deconv_bias", [inpu_dim], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x_input):
		batch_size, width, channels = x_input.get_shape().as_list()
		x = tf.expand_dims(x_input, [1])
		act_x = tf.nn.conv2d_transpose(x, self.filter, output_shape = [batch_size, 1, width * self.up_factor, channels], strides=self.up_factor)
		act_x = tf.squeeze(act_x, [1])
		return act_x + self.bias

#This is sub-pixel deconvolution.
class Learned_SubPixel(object):
	def __init__(self, scope, input_dim, up_factor):
		self.up_factor = up_factor
		self.var_scope = "learn_subpix_%s" %(scope)

		with tf.variable_scope(self.var_scope):
			self.expander = Conv_1x1("subpix_up", input_dim, input_dim * up_factor)

	def sub_pixel_1D(self, act, expansion = 2):
		up_sampled = tf.transpose(acts, [2, 1, 0])
		up_sampled = tf.batch_to_space_nd(up_sampled, [expansion], [[0, 0]])
		return tf.transpose(up_sampled, [2, 1, 0])

	def activated_on(self, x_input):
		up_sampled = self.sub_pixel_1D(self.expander.activated_on(x_input), expansion = self.up_factor)
		return up_sampled

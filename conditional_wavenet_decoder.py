import tensorflow as tf
import numpy as np

from layers import *
from BN_layers import *

class Causal_Conv(object):
	def __init__(self, var_scope, filter_width, in_channels, out_channels, dilation):
		self.var_scope = var_scope
		self.filter_width = filter_width
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.dilation = dilation

		with tf.variable_scope(self.var_scope):
			self.filter = tf.get_variable("filter", [self.filter_width, self.in_channels, self.out_channels], tf.float32, tf.random_normal_initializer(0.0, 0.05))
			self.bias = tf.get_variable("bias", [1, self.out_channels], tf.float32, tf.constant_initializer(0.0))

	def activated_on(self, x):
		with tf.variable_scope(self.var_scope):
			left_pad = self.dilation * (self.filter_width - 1)
			padded = tf.pad(x, [[0, 0], [left_pad, 0], [0, 0]])
			conv_act = tf.nn.convolution(padded, self.filter, padding="VALID", dilation_rate = [self.dilation])
			return conv_act + self.bias

class Unconditional_Wave_Block(object):
	def __init__(self, prefix, filter_width, conv_in_channels, conv_out_channels, skip_channels, dilation = 1, last_layer = False, z_channels = 0, use_skip = True):
		self.use_dense = True
		self.use_skip = use_skip
		self.glu = True

		self.x_filter = Causal_Conv("%s_x_filter" %(prefix), 2, conv_in_channels, conv_out_channels, dilation = dilation)

		if self.glu:
			self.x_gate = Causal_Conv("%s_x_gate" %(prefix), 2, conv_in_channels, conv_out_channels, dilation = dilation)

		self.dense = Conv_1x1("%s_dense" %(prefix), conv_out_channels, conv_out_channels)
		if self.use_skip:
			self.skip = Conv_1x1("%s_skip" %(prefix), conv_out_channels, skip_channels)

	def activated_on(self, x, h = None):
		x_filter = self.x_filter.activated_on(x)
		if self.glu:
			x_gate = self.x_gate.activated_on(x)

		if self.glu:
			out = x_filter * tf.nn.sigmoid(x_gate) #Gated Linear Unit
		else:
			out = tf.nn.relu(x_filter)

		dense = self.dense.activated_on(out)
		if self.use_skip:
			skip = self.skip.activated_on(out)
		else:
			skip = None

		return tf.nn.relu(x + dense), skip

class Spatial_Conditional_Wave_Block(object):
	def __init__(self, prefix, is_training, filter_width, conv_in_channels, conv_out_channels, skip_channels, dilation = 1, last_layer = False, z_channels = 0, use_skip = True):
		self.use_dense = True
		self.use_skip = use_skip
		self.glu = True

		self.x_filter = Causal_Conv("%s_x_filter" %(prefix), 2, conv_in_channels, conv_out_channels, dilation = dilation)
		self.x_gate = Causal_Conv("%s_x_gate" %(prefix), 2, conv_in_channels, conv_out_channels, dilation = dilation)

		self.x_gamma = tf.get_variable("%s_x_gamma" %(prefix), [1, conv_out_channels], tf.float32, tf.constant_initializer(1.0))
		self.x_beta = tf.get_variable("%s_x_beta" %(prefix), [1, conv_out_channels], tf.float32, tf.constant_initializer(0.0))

		self.h_filter = BN_Conv_1x1("%s_h_filter" %(prefix), is_training, z_channels, conv_out_channels)
		self.h_gate = BN_Conv_1x1("%s_h_gate" %(prefix), is_training, z_channels, conv_out_channels)

		self.z_gamma = tf.get_variable("%s_z_gamma" %(prefix), [1, conv_out_channels], tf.float32, tf.constant_initializer(1.0))
		self.z_beta = tf.get_variable("%s_z_beta" %(prefix), [1, conv_out_channels], tf.float32, tf.constant_initializer(0.0))

		self.dense = Conv_1x1("%s_dense" %(prefix), conv_out_channels, conv_out_channels)
		if self.use_skip:
			self.skip = Conv_1x1("%s_skip" %(prefix), conv_out_channels, skip_channels)

	def activated_on(self, x, h):
		x_filter = self.x_filter.activated_on(x)
		x_gate = self.x_gate.activated_on(x)
		h_filter = self.h_filter.activated_on(h)
		h_gate = self.h_gate.activated_on(h)

		out = (x_filter + h_filter) * tf.sigmoid(x_gate + h_gate)

		batch_size, seq_len, channels = x.get_shape().as_list()

		rs = tf.reshape(out, [batch_size, seq_len, 16, -1]) #make 16 groups of features. 32 channels each.
		mean, var = tf.nn.moments(rs, [3], keep_dims = True)
		rs = (rs - mean) / tf.sqrt(var + 1e-8)

		bn = self.x_gamma * tf.reshape(rs, [batch_size, seq_len, channels]) + self.x_beta

		dense = self.dense.activated_on(out)

		rs = tf.reshape(dense, [batch_size, seq_len, 16, -1])
		mean, var = tf.nn.moments(rs, [3], keep_dims = True)
		rs = (rs - mean) / tf.sqrt(var + 1e-8)

		dense = self.z_gamma * tf.reshape(rs, [batch_size, seq_len, channels]) + self.z_beta

		if self.use_skip:
			skip = self.skip.activated_on(out)
		else:
			skip = None

		return x + dense, skip

class Wavenet_Decoder(object):
	def __init__(self, batch_size, prot_len, discrete_dims = 4, do_embed = True, z_channels = None, dilations = [1, 2, 4, 8, 16], residual_channels = 256, dilation_channels = 256, skip_channels = 512, use_skip = False):
		self.batch_size = batch_size
		if do_embed:
			self.aa_embedding_size = 64
		self.discrete_dims = discrete_dims
		self.do_embed = do_embed
		self.z_channels = z_channels
		self.prot_len = prot_len #going to require this even though it's a bit annoying for the positional arguments.

		self.position_dim = 4
		self.do_position = True

		self.use_skip = use_skip
		self.residual_channels = residual_channels
		self.dilation_channels = dilation_channels
		self.skip_channels = skip_channels

		if self.use_skip:
			self.model_output_dim = self.skip_channels
		else:
			self.model_output_dim = self.residual_channels

		self.filter_width = 2
		self.dilations = dilations

		if z_channels is None:
			self.block_class = Unconditional_Wave_Block
		else:
			self.block_class = Spatial_Conditional_Wave_Block
		self.vars = self.create_variables()

	def create_variables(self):
		var = {}
		with tf.variable_scope("wavenet"):
			with tf.variable_scope("input_layer"):
				if self.do_embed:
					initial_channels = self.aa_embedding_size
					var["aa_embed"] = Conv_1x1("aa_embedding", self.discrete_dims, self.aa_embedding_size)
				else:
					initial_channels = self.discrete_dims

				var["position_low_dim"] = tf.get_variable("position_low_dim", [1, self.prot_len, self.position_dim], tf.float32, tf.random_normal_initializer(0.0, 0.05))
				var["low_dim_pos_up"] = Conv_1x1("low_dim_pos_up", self.position_dim, initial_channels)

				var["input_conv"] = Causal_Conv("input_conv", 2, initial_channels, self.residual_channels, dilation = 1)

		with tf.variable_scope("dilated_convolutions"):
			var["dilated_convolutions"] = []
			for (layer_index, dilation) in enumerate(self.dilations):
				next_layer = self.block_class("wavenet_%i" %(layer_index), self.filter_width, self.residual_channels, self.dilation_channels, self.skip_channels, dilation = dilation, z_channels = self.z_channels, use_skip = self.use_skip)
				var["dilated_convolutions"].append(next_layer)

		with tf.variable_scope("post_process"):
			var["post_process_one"] = Conv_1x1("post_process_one", self.model_output_dim, self.model_output_dim)
			var["post_process_two"] = Conv_1x1("post_process_two", self.model_output_dim, self.discrete_dims)

		return var

	def run_conv(self, batch, h = None):
		skip_outputs = []

		if self.do_embed:
			embedded_batch = self.vars["aa_embed"].activated_on(batch)
		else:
			embedded_batch = batch

		if self.do_position:
			embedded_batch += self.vars["low_dim_pos_up"].activated_on(self.vars["position_low_dim"])

		cur_act = self.vars["input_conv"].activated_on(embedded_batch)

		for layer in self.vars["dilated_convolutions"]:
			cur_act, skip = layer.activated_on(cur_act, h)
			skip_outputs.append(skip)

		if self.use_skip:
			return sum(skip_outputs), cur_act
		else:
			return None, cur_act

	def run_post(self, final_act):
		dense_act = self.vars["post_process_one"].activated_on(final_act)
		dense_act = tf.nn.relu(dense_act)
		nuc_hidden = self.vars["post_process_two"].activated_on(dense_act)
		return nuc_hidden

	def activated_on(self, batch, h = None):
		#batch = self.one_hot(batch)
		if self.use_skip:
			net_out, _ = self.run_conv(batch, h = h)
		else:
			_, net_out = self.run_conv(batch, h = h)

		#if h is None:
			#seq_pred = self.run_post(net_out)
			#return seq_pred
		#else:
		seq_pred = self.run_post(net_out)
		return seq_pred

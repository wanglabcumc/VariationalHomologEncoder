import tensorflow as tf
import numpy as np

from layers import *
from BN_layers import *

class Dilated_Block(object):
	def __init__(self, prefix, is_training, filter_width, conv_in_channels, conv_out_channels, skip_channels, dilation, clust_size = None, use_skip = True):
		self.use_dense = True
		self.use_dropout = False
		self.use_skip = use_skip
		self.glu = True
		self.clust_size = clust_size

		self.x_filter = BN_Conv("%s_x_filter" %(prefix), is_training, filter_width, conv_in_channels, conv_out_channels, dilation = dilation)
		if self.glu:
			self.x_gate = BN_Conv("%s_x_gate" %(prefix), is_training, filter_width, conv_in_channels, conv_out_channels, dilation = dilation)

		self.dense = BN_Conv_1x1("%s_dense" %(prefix), is_training, conv_out_channels, conv_out_channels)
		if self.use_skip:
			self.skip = BN_Conv_1x1("%s_skip" %(prefix), is_training, conv_out_channels, skip_channels)

	def activated_on(self, x):
		x_filter = self.x_filter.activated_on(x)
		if self.glu:
			x_gate = self.x_gate.activated_on(x)

		if self.glu:
			out = x_filter * tf.sigmoid(x_gate)
		else:
			out = tf.nn.relu(x_filter)

		dense = self.dense.activated_on(out)
		if self.use_skip:
			skip = self.skip.activated_on(out)
		else:
			skip = None

		return x + dense, skip

class Dilated_Encoder(object):
	def __init__(self, name, is_training, batch_size, max_seq_len, channels, discrete_dims = 22, embedding_size = 32, do_embed = True, use_skip = False):
		self.batch_size = batch_size
		self.var_scope = name
		self.max_seq_len = max_seq_len
		self.is_training = is_training

		self.positional_encoding = True
		self.embedding_size = embedding_size
		self.discrete_dims = discrete_dims
		self.position_embedding_size = self.discrete_dims
		self.do_embed = do_embed

		self.use_skip = use_skip
		self.residual_channels = channels
		self.dilation_channels = channels
		self.skip_channels = channels

		self.filter_width = 3
		self.dilations = [1, 3, 9, 27]

		self.model_output_dim = self.skip_channels if self.use_skip else self.residual_channels
		self.block_class = Dilated_Block
		self.vars = self.create_variables()

	def create_variables(self):
		var = {}
		with tf.variable_scope(self.var_scope):
			with tf.variable_scope("wavenet_encoder"):
				if self.do_embed:
					initial_channels = self.embedding_size
					var["seq_embed"] = Conv_1x1("seq_embed", self.discrete_dims, self.embedding_size)
				else:
					initial_channels = self.discrete_dims

				if self.positional_encoding:
					var["position_encoder"] = tf.get_variable("enc_position_encoder", [1, self.max_seq_len, self.position_embedding_size], tf.float32, tf.random_normal_initializer(0.0, 0.05))
					var["position_1x1"] = Conv_1x1("pos_embed", self.position_embedding_size, initial_channels)
				var["input_conv"] = BN_Conv("input_conv", self.is_training, 3, initial_channels, self.residual_channels, dilation = 1)

		with tf.variable_scope("dilated_convolutions"):
			var["dilated_convolutions"] = []
			for (layer_index, dilation) in enumerate(self.dilations):
				next_layer = self.block_class("encoding_wavenet_%i" %(layer_index), self.is_training, self.filter_width, self.residual_channels, self.dilation_channels, self.skip_channels, dilation = dilation, use_skip = self.use_skip)
				var["dilated_convolutions"].append(next_layer)

		return var

	def run_conv(self, batch):
		skip_outputs = []
		if self.do_embed:
			embedded_batch = self.vars["seq_embed"].activated_on(batch)
		else:
			embedded_batch = batch

		if self.positional_encoding:
			embedded_batch += self.vars["position_1x1"].activated_on(self.vars["position_encoder"])

		cur_act = self.vars["input_conv"].activated_on(embedded_batch)

		for layer in self.vars["dilated_convolutions"]:
			cur_act, skip = layer.activated_on(cur_act)
			skip_outputs.append(skip)

		if self.use_skip:
			return sum(skip_outputs), cur_act
		else:
			return None, cur_act

	def activated_on(self, batch):
		if self.use_skip:
			net_out, _ = self.run_conv(batch)
		else:
			_, net_out = self.run_conv(batch)

		return net_out

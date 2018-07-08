import tensorflow as tf
import numpy as np

from upsampling import *
from rnn import *
from layers import *

"""Sample-level logic. This is machinery used at lowest level where predictions of individual next-characters are being made."""

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

class Conditional_Wave_Block(object):
	def __init__(self, prefix, filter_width, conv_in_channels, conv_out_channels, skip_channels, dilation = 1, h_channels = None):
		self.use_dense = True
		self.use_skip = False #very much doubt skip connections will ever be useful for a two layer network.
		self.use_glu = True
		self.dilation = dilation
		self.output_dim = skip_channels if self.use_skip else conv_out_channels

		self.x_filter = Causal_Conv("%s_x_filter" %(prefix), 2, conv_in_channels, conv_out_channels, dilation = dilation)
		self.x_gate = Causal_Conv("%s_x_gate" %(prefix), 2, conv_in_channels, conv_out_channels, dilation = dilation)

		self.h_filter = Conv_1x1("%s_h_filter" %(prefix), h_channels, conv_out_channels)
		self.h_gate = Conv_1x1("%s_h_gate" %(prefix), h_channels, conv_out_channels)

		self.dense = Conv_1x1("%s_dense" %(prefix), conv_out_channels, conv_out_channels)
		if self.use_skip:
			self.skip = Conv_1x1("%s_skip" %(prefix), conv_out_channels, skip_channels)

	def activated_on(self, x, h):
		x_filter = self.x_filter.activated_on(x)
		x_gate = self.x_gate.activated_on(x)
		h_filter = self.h_filter.activated_on(h)
		h_gate = self.h_gate.activated_on(h)

		out = (x_filter + h_filter) * tf.sigmoid(x_gate + h_gate)

		dense = self.dense.activated_on(out)

		if self.use_skip:
			skip = self.skip.activated_on(out)
		else:
			skip = None

		return x + dense, skip

#Each dimension sees last 4 aa, even at borders of frames. Also sees upper-tier conditioning. Should be faster than base-RNN.
class Sample_Level_Wave(object):
	def __init__(self, scope, frame_size, hidden_size, input_channels, q_channels = 22):
		self.input_channels = input_channels
		self.var_scope = "sample_level_wave_%s" %(scope)
		self.filter_width = 2

		self.residual_channels = hidden_size
		self.dilation_channels = hidden_size
		self.h_channels = hidden_size
		self.use_skip = False

		self.skip_channels = 0
		self.depth = 1
		self.dilations = [1, 2] * self.depth

		with tf.variable_scope(self.var_scope):
			self.input_conv = Conv_1x1("input_conv", input_channels, hidden_size)

			self.dilated_convolutions = []
			for (layer_index, dilation) in enumerate(self.dilations):
				next_layer = Conditional_Wave_Block("condi_wave_%i" %(layer_index), self.filter_width,
														self.residual_channels, self.dilation_channels, self.skip_channels,
														dilation = dilation, h_channels = self.h_channels)
				self.dilated_convolutions.append(next_layer)

			self.output_conv_one = Conv_1x1("output_conv_one", next_layer.output_dim, next_layer.output_dim)
			self.output_conv_two = Conv_1x1("output_conv_two", next_layer.output_dim, q_channels)

	def activated_on(self, batch, conditioning):
		skip_outputs = []

		cur_act = self.input_conv.activated_on(batch)

		for layer in self.dilated_convolutions:
			cur_act, skip = layer.activated_on(cur_act, conditioning)
			skip_outputs.append(skip)

		if self.use_skip:
			final_out = sum(skip_outputs)
		else:
			final_out = cur_act

		last_hid = self.output_conv_one.activated_on(final_out)
		last_hid = tf.nn.relu(last_hid)
		return self.output_conv_two.activated_on(last_hid)

#Each dimension sees a maximum of 3 preceding aas, but doesn't skip over frame borders. More closely equivalent to SampleRNN definition. Also has upper tier conditioning.
#We'll code this later as I'm not sure there's much sense to the idea.
class Sample_Level_DWave(object):
	def __init__(self):
		pass

#Each dimension is independent given upper-tier conditioning.
class Sample_Level_MLP(object):
	def __init__(self, scope, frame_size, hidden_size, input_channels, q_channels = 22):
		self.input_channels = input_channels
		self.var_scope = "sample_level_MLP_%s" %(scope)

		with tf.variable_scope(self.var_scope):
			self.input_conv = Conv_1x1("input_conv", input_channels, hidden_size)
			self.hidden_conv = Conv_1x1("hidden_conv", hidden_size, hidden_size)
			self.output_conv = Conv_1x1("output_conv", hidden_size, q_channels)

	def activated_on(self, prev_samples, upper_tier_conditioning):
		input_on_prev = self.input_conv.activated_on(prev_samples)
		h = tf.nn.relu(self.hidden_conv.activated_on(input_on_prev + upper_tier_conditioning))
		pred = self.output_conv.activated_on(h)
		return pred

"""Frame-Level work... these are higher up in hierarchy and use RNNs."""

class Frame_Level_RNN(object):
	def __init__(self, scope, input_channels, frame_size, n_rnn, hidden_size, batch_size, max_prot_len, forget_bias = 1.0, up_factor = 4):
		self.frame_size = frame_size
		self.hidden_size = hidden_size
		self.var_scope = "frame_level_RNN_%s" %(scope)

		with tf.variable_scope(self.var_scope):
			self.embedder = Conv_1x1("input_embedder", input_channels, hidden_size)
			self.rnn = My_RNN("frame_rnn", hidden_size, hidden_size, batch_size, max_prot_len, num_layers = n_rnn, forget_bias = forget_bias) #there's probably some more arguments to add in future but w/e for now.
			self.up_sample = Learned_Upsample("frame_upsample", hidden_size, up_factor)

	def embed(self, x_input):
		embedded = self.embedder.activated_on(x_input)
		return embedded

	def activated_on(self, x_in, upper_conditioning, last_hidden):
		#embedded = self.embed(x_in)
		with tf.variable_scope(self.var_scope):
			rnn_output, last_hidden = self.rnn.activated_on(x_in + upper_conditioning, initial_state = last_hidden)
			#last_hidden = tf.expand_dims(last_hidden, [1])
			up_sampled = self.up_sample.activated_on(tf.expand_dims(last_hidden[1], [1])) # + upper_conditioning)
		return up_sampled, rnn_output, last_hidden

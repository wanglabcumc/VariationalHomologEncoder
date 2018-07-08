import numpy as np
import tensorflow as tf

import collections

from tensorflow.python.ops.nn import rnn_cell

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))

class LSTMStateTuple(_LSTMStateTuple):
	__slots__ = ()

	@property
	def dtype(self):
		(c, h) = self
		if not c.dtype == h.dtype:
			raise TypeError("Inconsistent internal state: %s vs %s" %(str(c.dtype), str(h.dtype)))
		return c.dtype

class LSTM_LN_Cell(tf.contrib.rnn.RNNCell):
	def __init__(self, num_units, num_inputs = 22, forget_bias = 1.0, activation = tf.tanh, mul_int = False, layer_norm = False):
		self._num_units = num_units
		self._num_inputs = num_inputs
		self._forget_bias = forget_bias
		self._activation = activation
		self._state_is_tuple = True
		self.epsilon = 1e-6 #small number for normalizer.
		self._MI = mul_int
		self._LN = layer_norm

	@property
	def input_size(self):
		return self._num_inputs

	@property
	def output_size(self):
		return self._num_inputs

	@property
	def state_size(self):
		return (LSTMStateTuple(self._num_units, self._num_units))

	def normalize_acts(self, activations):
		mu, var = tf.nn.moments(activations, [0])
		std_dev = tf.sqrt(var) + self.epsilon
		normalized = (activations - mu) / std_dev
		return normalized

	def __call__(self, inputs, state, timestep=0):
		with tf.variable_scope("LSTM"):
			c, h = state
			concat_bias = tf.get_variable("concat_bias", [self._num_units * 4], tf.float32, tf.constant_initializer(0.0))
			if self._MI:
				mi_alpha = tf.get_variable("MI_alpha", [self._num_units * 4], tf.float32, tf.constant_initializer(1.0))

			with tf.variable_scope("input_weight_matrix"):
				W_x = tf.get_variable("W_x", [self._num_inputs, self._num_units * 4], tf.float32, tf.contrib.layers.variance_scaling_initializer()) #MSRA/He-init.
				if self._MI:
					mi_beta_x = tf.get_variable("MI_beta_x", [self._num_units * 4], tf.float32, tf.constant_initializer(0.5))

				if self._LN:
					gamma_wx = tf.get_variable("gamma_wx", [self._num_units * 4], tf.float32, tf.constant_initializer(1.0))
					beta_wx = tf.get_variable("beta_wx", [self._num_units * 4], tf.float32, tf.constant_initializer(0.0))
					input_act = tf.matmul(inputs, W_x)
					input_norm = self.normalize_acts(input_act)
					Wx_act = gamma_wx * input_norm + beta_wx
				else:
					Wx_act = tf.matmul(inputs, W_x)

			with tf.variable_scope("hidden_hidden_matrix"):
				W_h = tf.get_variable("W_h", [self._num_units, self._num_units * 4], tf.float32, tf.orthogonal_initializer())
				if self._MI:
					mi_beta_h = tf.get_variable("MI_beta_h", [self._num_units * 4], tf.float32, tf.constant_initializer(0.5))

				if self._LN:
					gamma_wh = tf.get_variable("gamma_wh", [self._num_units * 4], tf.float32, tf.constant_initializer(1.0))
					beta_wh = tf.get_variable("beta_wh", [self._num_units * 4], tf.float32, tf.constant_initializer(0.0))
					h_act = tf.matmul(h, W_h)
					h_norm = self.normalize_acts(h_act)
					Wh_act = gamma_wh * h_norm + beta_wh
				else:
					Wh_act = tf.matmul(h, W_h)

			if self._MI:
				concat_acts = mi_alpha * Wh_act * Wx_act + mi_beta_h * Wh_act + mi_beta_x * Wx_act + concat_bias
			else:
				concat_acts = Wh_act + Wx_act + concat_bias

			f_t, i_t, o_t, g_t = tf.split(concat_acts, num_or_size_splits = 4, axis = 1)
			f_t += self._forget_bias #I think f is the forget gate.

			c_t = tf.sigmoid(f_t) * c + tf.sigmoid(i_t) * self._activation(g_t)

			if self._LN:
				with tf.variable_scope("context_normalizing"):
					gamma_c = tf.get_variable("gamma_c", [self._num_units], tf.float32, tf.constant_initializer(1.0))
					beta_c = tf.get_variable("beta_c", [self._num_units], tf.float32, tf.constant_initializer(0.0))
					c_norm = self.normalize_acts(c_t)
					c_t = gamma_c * c_norm + beta_c

			h_t = tf.sigmoid(o_t) * self._activation(c_t)
			#The LayerNorm Tensorflow implementation does just normalize context and carry on.
			#I think the suggestion that we normalize c_t but then pass original on is a typo in LN paper...

			return h_t, LSTMStateTuple(c_t, h_t)

class GRU_LN_Cell(tf.contrib.rnn.RNNCell):
	def __init__(self, num_units, num_inputs = 22, forget_bias = 1.0, activation = tf.tanh, mul_int = False):
		self._num_units = num_units
		self._num_inputs = num_inputs
		self._forget_bias = forget_bias
		self._activation = activation
		#self._state_is_tuple = True
		self.epsilon = 1e-8
		self._MI = False

	@property
	def input_size(self):
		return self._num_inputs

	@property
	def output_size(self):
		return self._num_inputs

	@property
	def state_size(self):
		return (self._num_units, self._num_units)

	def normalize_acts(self, activations):
		mu, var = tf.nn.moments(activations, [0])
		std_dev = tf.sqrt(var) + self.epsilon
		normalized = (activations - mu) / std_dev
		return normalized

	def __call__(self, inputs, state, timestep = 0):
		with tf.variable_scope("LN_GRU"):
			c, h = state
			concat_bias = tf.get_variable("concat_bias", [self._num_units * 2], tf.float32, tf.constant_initializer(0.0))
			if self._MI:
				mi_alpha = tf.get_variable("MI_alpha", [self._num_units * 2], tf.float32, tf.constant_initializer(1.0))

			with tf.variable_scope("input_weight_matrix"):
				W_x = tf.get_variable("W_x", [self._num_inputs, self._num_units * 2], tf.float32, tf.contrib.layers.variance_scaling_initializer())
				gamma_wx = tf.get_variable("gamma_wx", [self._num_units * 2], tf.float32, tf.constant_initializer(1.0))
				beta_wx = tf.get_variable("beta_wx", [self._num_units * 2], tf.float32, tf.constant_initializer(0.0))
				if self._MI:
					mi_beta_x = tf.get_variable("MI_beta_x", [self._num_units * 2], tf.float32, tf.constant_initializer(0.5))

				input_act = tf.matmul(inputs, W_x)
				input_norm = self.normalize_acts(input_act)
				Wx_act = gamma_wx * input_norm + beta_wx

			with tf.variable_scope("hidden_hidden_matrix"):
				W_h = tf.get_variable("W_h", [self._num_units, self._num_units * 2], tf.float32, tf.orthogonal_initializer())
				gamma_wh = tf.get_variable("gamma_wh", [self._num_units * 2], tf.float32, tf.constant_initializer(1.0))
				beta_wh = tf.get_variable("beta_wh", [self._num_units * 2], tf.float32, tf.constant_initializer(0.0))
				if self._MI:
					mi_beta_h = tf.get_variable("MI_beta_h", [self._num_units * 2], tf.float32, tf.constant_initializer(0.5))

				h_act = tf.matmul(h, W_h)
				h_norm = self.normalize_acts(h_act)
				Wh_act = gamma_wh * h_norm + beta_wh

			if self._MI:
				concat_acts = mi_alpha * Wh_act * Wx_act + mi_beta_x * Wx_act + mi_beta_h * Wh_act + concat_bias
			else:
				concat_acts = Wh_act + Wx_act + concat_bias

			z_t, r_t = tf.split(concat_acts, num_or_size_splits = 2, axis = 1)
			z_t += self._forget_bias
			z_t = tf.sigmoid(z_t)

			with tf.variable_scope("candidate_matrix"):
				U_h = tf.get_variable("U_h", [self._num_units, self._num_units], tf.float32, tf.orthogonal_initializer())
				U_x = tf.get_variable("U_x", [self._num_inputs, self._num_units], tf.float32, tf.contrib.layers.variance_scaling_initializer())

				gamma_uh = tf.get_variable("gamma_uh", [self._num_units], tf.float32, tf.constant_initializer(1.0))
				beta_uh = tf.get_variable("beta_uh", [self._num_units], tf.float32, tf.constant_initializer(0.0))
				gamma_ux = tf.get_variable("gamma_ux", [self._num_units], tf.float32, tf.constant_initializer(1.0))
				beta_ux = tf.get_variable("beta_ux", [self._num_units], tf.float32, tf.constant_initializer(0.0))
				if self._MI:
					mi_alpha_hat = tf.get_variable("mi_alpha_hat", [self._num_units], tf.float32, tf.constant_initializer(1.0))
					mi_beta_hat_h = tf.get_variable("mi_beta_hat_h", [self._num_units], tf.float32, tf.constant_initializer(0.5))
					mi_beta_hat_x = tf.get_variable("mi_beta_hat_x", [self._num_units], tf.float32, tf.constant_initializer(0.5))
					mi_beta_hat = tf.get_variable("mi_beta_hat", [self._num_units], tf.float32, tf.constant_initializer(0.0))

				Ux_hat = tf.matmul(inputs, U_x)
				Ux_hat_norm = self.normalize_acts(Ux_hat)
				Ux_hat_act = gamma_ux * Ux_hat_norm + beta_ux

				Uh_hat = tf.matmul(h, U_h)
				Uh_hat_norm = self.normalize_acts(Uh_hat)
				Uh_hat_act = gamma_uh * Uh_hat_norm + beta_uh

				cand_act = tf.sigmoid(r_t) * Uh_hat_act

				if self._MI:
					h_proposed = self._activation(mi_alpha_hat * Ux_hat_act * cand_act + mi_beta_hat_h * cand_act + mi_beta_hat_x * Ux_hat_act + mi_beta_hat)
				else:
					h_proposed = self._activation(Ux_hat_act + cand_act)

				h_t = z_t * h + (1. - z_t) * h_proposed

			return h_t, (h_t, h_t)

class My_RNN(object):
	def __init__(self, scope, input_size, hidden_size, batch_size, max_len, num_layers = 1, num_transitions = 1, rnn_type = "LN_LSTM", forget_bias = 1.0):
		self.batch_size = batch_size
		self.max_len = max_len
		self.input_size = input_size
		self.scope = scope

		with tf.variable_scope(self.scope):
			# GRU_LN_Cell(hidden_size, hidden_size, forget_bias = 1.0, activation = tf.tanh, mul_int = True) #
			self.encoding_cell = LSTM_LN_Cell(hidden_size, hidden_size, forget_bias = forget_bias, activation = tf.tanh, mul_int = True) #tf.nn.rnn_cell.GRUCell(hidden_size)
			self.dummy_input = tf.placeholder(shape = [self.batch_size, self.max_len, self.input_size], dtype = tf.float32)
			output, hidden = tf.nn.dynamic_rnn(self.encoding_cell, self.dummy_input, dtype = tf.float32)

	def activated_on(self, inputs, initial_state = None):
		with tf.variable_scope(self.scope, reuse = True):
			output, hidden = tf.nn.dynamic_rnn(self.encoding_cell, inputs, initial_state = initial_state, dtype = tf.float32)
		return output, hidden

import tensorflow as tf
import numpy as np

from frames import *
from upsampling import *
from BN_layers import BN_Dense

class Sample_RNN(object):
	def __init__(self, is_training, frame_sizes, n_rnn, hidden_size, learn_h0, input_channels, batch_size, max_prot_len, global_conditioning = False, global_latent_size = 512, up_factor = 4):
		self.hidden_size = hidden_size
		self.input_channels = input_channels
		self.batch_size = batch_size
		self.max_prot_len = max_prot_len
		self.frame_widths = frame_sizes
		self.global_conditioning = global_conditioning
		self.up_factor = up_factor

		#print "Curiously enough, I htink batch size is", batch_size

		self.frame_level_rnns = []
		self.frame_zeros = {}
		self.global_dense = {}

		tier_count = 0
		for frame_size in frame_sizes:
			self.frame_zeros[tier_count] = (tf.get_variable("initial_c_%i" %(tier_count), [1, hidden_size], tf.float32, tf.constant_initializer(0.0)),
											tf.get_variable("initial_h_%i" %(tier_count), [1, hidden_size], tf.float32, tf.constant_initializer(0.0)))
			self.frame_level_rnns.append(Frame_Level_RNN("tier_%i" %(tier_count + 1), self.input_channels, frame_size, n_rnn, hidden_size, batch_size, max_prot_len, up_factor = self.up_factor))

			if self.global_conditioning:
				self.global_dense[tier_count] = BN_Dense("global_dense_%i" %(tier_count), is_training, global_latent_size, self.up_factor * hidden_size)

			tier_count += 1

		if self.global_conditioning:
			self.global_dense[-1] = BN_Dense("global_dense_top", is_training, global_latent_size, self.up_factor * hidden_size)

		self.sample_level_mlp = Sample_Level_Wave("tier_0", frame_sizes[-1], hidden_size, input_channels)

	def initial_state(self, state_var):
		state_c, state_h = state_var
		start_c = tf.reshape(tf.tile(state_c, [1, self.batch_size]), [self.batch_size, self.hidden_size])
		start_h = tf.reshape(tf.tile(state_h, [1, self.batch_size]), [self.batch_size, self.hidden_size])
		return LSTMStateTuple(start_c, start_h)

	def activated_on(self, x_in, global_conditioning = None):
		input_letters = x_in

		if not self.global_conditioning:
			last_upper_conditions = tf.zeros([self.batch_size, self.up_factor, self.hidden_size])
		else:
			last_upper_conditions = self.global_dense[-1].activated_on(global_conditioning)
			last_upper_conditions = tf.reshape(last_upper_conditions, [self.batch_size, self.up_factor, self.hidden_size])

		frame_tier = 0
		for (tier_width, tier_rnn) in zip(self.frame_widths, self.frame_level_rnns):
			x_in = tier_rnn.embed(input_letters)
			num_frames = self.max_prot_len // tier_width
			frame_inputs = tf.split(x_in, num_or_size_splits = num_frames, axis = 1)
			upper_conditioning = tf.split(last_upper_conditions, num_or_size_splits = num_frames, axis = 1)

			last_hidden = self.initial_state(self.frame_zeros[frame_tier])

			layer_outs = []
			f_in_count = 0

			for f_in in frame_inputs[:-1]:
				upsampled, out, hidden = tier_rnn.activated_on(f_in, upper_conditioning[f_in_count], last_hidden)
				last_hidden = hidden
				layer_outs.append(upsampled)
				f_in_count += 1

			if self.global_conditioning:
				z_details = self.global_dense[frame_tier].activated_on(global_conditioning)
				z_details = tf.reshape(z_details, [self.batch_size, self.up_factor, self.hidden_size])
				last_upper_conditions = tf.concat([z_details, tf.concat(layer_outs, axis = 1)], axis = 1)
			else:
				last_upper_conditions = tf.concat([tf.zeros([self.batch_size, self.up_factor, self.hidden_size]), tf.concat(layer_outs, axis = 1)], axis = 1)
			frame_tier += 1

		logit_predictions = self.sample_level_mlp.activated_on(input_letters, last_upper_conditions)
		return logit_predictions

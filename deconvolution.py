import tensorflow as tf
import numpy as np

from BN_layers import *
from layers import *

class Identity_1x1(object):
	def __init__(self, name, in_channels, out_channels):
		self.name = name
		self.in_channels = in_channels
		self.out_channels = out_channels

	def activated_on(self, x, expansion = 2):
		bs = x.shape[0]
		return tf.tile(x, [1, 1, expansion])

def sub_pixel_1D(conv_1x1_acts, expansion = 2):
	up_sampled = tf.transpose(conv_1x1_acts, [2, 1, 0])
	up_sampled = tf.batch_to_space_nd(up_sampled, [expansion], [[0, 0]])
	return tf.transpose(up_sampled, [2, 1, 0])

class SP_Deconv_256(object):
	def __init__(self, name, is_training, batch_size, channels):
		self.batch_size = batch_size
		self.channels = channels
		self.use_positional = False
		self.var_scope = name

		self.expands = []
		self.reses = []
		self.position_rands = []
		self.position_ups = []

		with tf.variable_scope(self.var_scope):
			#self.first_expand = BN_conv1d_dense("expand_start", self.channels, self.channels * 2)
			#self.first_res = Identity_1x1("residual_start", self.channels, self.channels * 2)

			for (expand, width) in zip(["128", "64", "32", "16", "8", "4", "2", "1"], [2, 4, 8, 16, 32, 64, 128, 256]):
				self.expands.append(BN_conv1d_dense("expand_z_%s" %(expand), is_training, self.channels, self.channels * 2))
				self.reses.append(Identity_1x1("residual_%s" %(expand), self.channels, self.channels * 2))

				if self.use_positional:
					self.position_rands.append(tf.get_variable("deconv_pos_embed_%s" %(expand), [1, width, 4]))
					self.position_ups.append(Conv_1x1("position_up_%s" %(expand), 4, channels))
				else:
					self.position_rands.append(None)
					self.position_ups.append(None)

	def activated_on(self, cur_act):
		for (expand, res, p_r, p_u) in zip(self.expands, self.reses, self.position_rands, self.position_ups):
			cur_res = sub_pixel_1D(res.activated_on(cur_act))
			cur_act = sub_pixel_1D(expand.activated_on(cur_act))

			cur_act = tf.nn.relu(cur_act + cur_res)

			if self.use_positional:
				cur_act += p_u.activated_on(p_r)
		return cur_act

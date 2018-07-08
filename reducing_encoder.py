import numpy as np
import os

import tensorflow as tf

from BN_layers import *
from layers import *

from utils import avg_pool_res

class Reducing_Encoder_243(object):
	def __init__(self, name, is_training, batch_size, channels = 256, filter_width = 3, strides = 3):
		self.batch_size = batch_size
		self.var_scope = name
		self.channels = channels
		self.filter_width = filter_width
		self.strides = strides
		self.glu = False

		self.up_f = []
		self.up_g = []
		self.avg_res = []

		with tf.variable_scope(self.var_scope):
			for input_size in [243, 81, 27, 9]:
				self.up_f.append(BN_Conv("up_f_%i" %(input_size), is_training, self.filter_width, self.channels, self.channels))
				if self.glu:
					self.up_g.append(BN_Conv("up_g_%i" %(input_size), is_training, self.filter_width, self.channels, self.channels))
				else:
					self.up_g.append(None)
				self.avg_res.append(avg_pool_res(self.filter_width, self.strides))

			self.up_dense_f = BN_Conv("final_act_f", is_training, 3, self.channels, self.channels)
			self.last_res = avg_pool_res(3, 3)
			if self.glu:
				self.up_dense_g = BN_Conv("final_act_g", is_training, 3, self.channels, self.channels)

	def activated_on(self, seq):
		cur_act = seq
		for (up_f, up_g, avg_res) in zip(self.up_f, self.up_g, self.avg_res):
			res_act = avg_res.activated_on(cur_act)
			f_act = up_f.activated_on(cur_act, strides = [self.strides])
			if self.glu:
				g_act = tf.sigmoid(up_g.activated_on(cur_act, strides=[self.strides]))
				cur_act = f_act * g_act + res_act
			else:
				cur_act = tf.nn.relu(f_act + res_act)

		if self.glu:
			dense_f = self.up_dense_f.activated_on(cur_act, strides = [3])
			dense_g = self.up_dense_g.activated_on(cur_act, strides = [3])
			dense_act = dense_f * tf.sigmoid(dense_g) + self.last_res.activated_on(cur_act)
		else:
			dense_f = self.up_dense_f.activated_on(cur_act, strides=[3]) + self.last_res.activated_on(cur_act) #why not just keep the residual going?
			dense_act = tf.nn.relu(dense_f + self.last_res.activated_on(cur_act))
		#dense_g = self.up_dense_g.activated_on(cur_act, strides=[5])
		#dense_act = dense_f * tf.sigmoid(dense_g)
		return tf.squeeze(dense_act, [1])

class Reducing_Encoder(object):
	def __init__(self, name, is_training, batch_size, channels = 256, filter_width = 2, strides = 2):
		self.batch_size = batch_size
		self.var_scope = name
		self.channels = channels
		self.filter_width = filter_width
		self.strides = strides
		self.glu = False

		self.up_f = []
		self.up_g = []
		self.avg_res = []

		with tf.variable_scope(self.var_scope):
			for input_size in [256, 128, 64, 32, 16, 8, 4]:
				self.up_f.append(BN_Conv("up_f_%i" %(input_size), is_training, self.filter_width, self.channels, self.channels))
				if self.glu:
					self.up_g.append(BN_Conv("up_g_%i" %(input_size), is_training, self.filter_width, self.channels, self.channels))
				else:
					self.up_g.append(None)
				self.avg_res.append(avg_pool_res(self.filter_width, self.strides))

			self.up_dense_f = BN_Conv("final_act_f", is_training, 2, self.channels, self.channels)
			self.last_res = avg_pool_res(2, 2)
			if self.glu:
				self.up_dense_g = BN_Conv("final_act_g", is_training, 2, self.channels, self.channels)

	def activated_on(self, seq):
		cur_act = seq
		for (up_f, up_g, avg_res) in zip(self.up_f, self.up_g, self.avg_res):
			res_act = avg_res.activated_on(cur_act)
			f_act = up_f.activated_on(cur_act, strides=[self.strides])
			if self.glu:
				g_act = tf.sigmoid(up_g.activated_on(cur_act, strides=[self.strides]))
				cur_act = f_act * g_act + res_act
			else:
				cur_act = tf.nn.relu(f_act + res_act)

		if self.glu:
			dense_f = self.up_dense_f.activated_on(cur_act, strides=[2])
			dense_g = self.up_dense_g.activated_on(cur_act, strides=[2])
			dense_act = dense_f * tf.sigmoid(dense_g) + self.last_res.activated_on(cur_act)
		else:
			dense_f = self.up_dense_f.activated_on(cur_act, strides=[2]) + self.last_res.activated_on(cur_act) #why not just keep the residual going?
			dense_act = tf.nn.relu(dense_f + self.last_res.activated_on(cur_act))
		#dense_g = self.up_dense_g.activated_on(cur_act, strides=[5])
		#dense_act = dense_f * tf.sigmoid(dense_g)
		return tf.squeeze(dense_act, [1])

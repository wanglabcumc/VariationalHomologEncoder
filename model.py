import time, datetime, random, os

import numpy as np
import tensorflow as tf

from HomoProtReader import HomoProtReader
from SameLenProtReader import NewProtReader
from layers import *
from BN_layers import BN_Dense

from dilated_encoder import Dilated_Encoder
from reducing_encoder import Reducing_Encoder_243
from sample_RNN_decoder import Sample_RNN

def basic_kl(mu, var):
	return -0.5 * tf.reduce_sum(1 + tf.log(var) - tf.square(mu) - var, 1)

def full_kl(mu, var):
	return -0.5 * (tf.log(var) - tf.square(mu) - var + 1)

def setup_log(prefix = "model_243_"):
	run_time = datetime.datetime.now()
	run_day = "%02d" %(run_time.day)
	run_month = "%02d" %(run_time.month)
	run_hour = "%02d" %(run_time.hour)
	run_minute = "%02d" %(run_time.minute)
	log_dir = prefix + run_month + run_day + "/"
	log_subdir = run_hour + "h_" + run_minute + "m/"

	if not os.path.isdir(log_dir):
		os.mkdir(log_dir)
	if not os.path.isdir(log_dir + log_subdir):
		os.mkdir(log_dir + log_subdir)

	return log_dir, log_subdir

class Homo_Model_243(object):
	def __init__(self, batch_size, max_seq_len, tab, model_hidden, prot_latent_rep, clust_size = 4, is_training = True):
		self.batch_size = batch_size
		self.max_seq_len = max_seq_len
		self.prot_latent_rep = prot_latent_rep
		self.clust_size = clust_size

		self.hash = tab

		self.prot_wave_in = Dilated_Encoder("prot_wave", is_training, self.batch_size, self.max_seq_len, channels = model_hidden, embedding_size = 64, do_embed = True, use_skip = True)
		self.prot_encoder = Reducing_Encoder_243("prot_encoder", is_training, self.batch_size, channels = model_hidden)
		self.prot_mu = BN_Dense("prot_mu", is_training, model_hidden, prot_latent_rep)
		self.prot_var = BN_Dense("prot_var", is_training, model_hidden, prot_latent_rep)

		self.decoder = Sample_RNN([81, 27, 9, 3], 1, model_hidden, True, 22, self.batch_size * (self.clust_size - 1),
									self.max_seq_len, global_conditioning = True, global_latent_size = prot_latent_rep, up_factor = 3)

	def one_hot(self, x):
		transformed = tf.one_hot(x, depth=22)
		transformed = tf.squeeze(transformed, [2])
		return tf.cast(transformed, tf.float32)

	def activated_on(self, rep_seq, homo_seq):
		prot_wave = self.prot_wave_in.activated_on(rep_seq)
		prot_code = self.prot_encoder.activated_on(prot_wave)

		prot_mu = self.prot_mu.activated_on(prot_code)
		prot_var = self.prot_var.activated_on(prot_code)
		prot_var = tf.nn.softplus(prot_var + 0.541325)

		prot_z = prot_mu + tf.sqrt(prot_var) * tf.random_normal(prot_mu.get_shape())

		prot_kl = basic_kl(prot_mu, prot_var)
		full_kl_loss = full_kl(prot_mu, prot_var)

		tf.summary.image("kl_viz", tf.expand_dims(tf.expand_dims(full_kl_loss, [0]), [3]))
		tf.summary.scalar("prot_KL", tf.reduce_mean(prot_kl, 0))

		triple_z = tf.reshape(tf.tile(prot_z, [1, 3]), [self.batch_size * 3, self.prot_latent_rep])
		recon = self.decoder.activated_on(homo_seq, triple_z)

		return recon, prot_kl

	def loss(self, catted_seq, kl_strength = 1.0):
		kl_strength = tf.minimum(kl_strength, 1.0)
		kl_strength = tf.maximum(0.0, kl_strength)

		tf.summary.scalar("kl_strength", kl_strength)

		reshaped_seq = tf.reshape(catted_seq, [self.batch_size, self.clust_size, self.max_seq_len, 1])
		rep_seq = reshaped_seq[:, 0, :, :] #get the first of the batch elements.
		homo_seq = tf.reshape(reshaped_seq[:, 1:, :, :], [self.batch_size * (self.clust_size - 1), self.max_seq_len, 1])

		rep_1h = self.one_hot(rep_seq)
		homo_1h = self.one_hot(homo_seq)

		rep_64 = tf.cast(tf.argmax(rep_1h, 2), tf.int64)
		tf.summary.text("rep", tf.slice(tf.reduce_join(self.hash.lookup(rep_64), reduction_indices=[1]), [0], [10]))

		homo_64 = tf.cast(tf.argmax(homo_1h, 2), tf.int64)
		tf.summary.text("homologs", tf.slice(tf.reduce_join(self.hash.lookup(homo_64), reduction_indices=[1]), [0], [10]))
		#Okay I feel more confident in this pipeline.

		homo_input = tf.pad(homo_1h, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

		seq_mask = tf.cast(tf.sign(tf.argmax(homo_1h, 2)), tf.float32)

		recon, kl_loss = self.activated_on(rep_1h, homo_input)

		raw_loss = tf.nn.softmax_cross_entropy_with_logits(labels = homo_1h, logits = recon)
		aa_loss = tf.reduce_mean(tf.reduce_sum(raw_loss * seq_mask, 1) / tf.reduce_sum(seq_mask, 1), 0)

		tf.summary.scalar("aa_loss", aa_loss)

		log_p_x_given_z = tf.reduce_sum(raw_loss * seq_mask, 1)
		loss = tf.reduce_sum(log_p_x_given_z, 0) + kl_strength * tf.reduce_sum(kl_loss, 0)

		tf.summary.scalar("opt_loss", loss)
		tf.summary.scalar("true_loss_per_aa", (tf.reduce_sum(log_p_x_given_z) + tf.reduce_sum(kl_loss)) / tf.reduce_sum(seq_mask))

		return loss / tf.reduce_sum(seq_mask)

def end_training(sess, coord, threads):
	coord.request_stop()
	coord.join(threads)
	sess.close()

def homo_train(log_dir, log_subdir, do_restore = False, file_to_restore = None):
	tf.reset_default_graph()

	batch_size = 64
	max_prot_len = 243

	coord = tf.train.Coordinator()
	clust_size = 4   #this includes the "representative" sequence.

	reader = HomoProtReader("train_SHAR90_clu60f75_l243.fa", coord, batch_size, max_prot_len = max_prot_len, set_length = clust_size)
	cat_batch = reader.dequeue(batch_size)

	int_to_char = {0: "-", 1: "A", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "K", 10: "L", 11: "M", 12: "N",
					13: "P", 14: "Q", 15: "R", 16: "S", 17: "T", 18: "V", 19: "W", 20: "Y", 21: "*"}
	my_hash = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([int_to_char[i] for i in range(22)]), default_value="?")
	#This is so you can see proteins and reconstructions, but Windows computers seem to have a hard time with it so maybe don't use.

	seq_len = max_prot_len

	with tf.variable_scope("model"):
		kl_strength = tf.get_variable("kl_strength", [], tf.float32, tf.constant_initializer(0.01), trainable = False)
		increase_kl = tf.assign_add(kl_strength, 1e-5)
		learning_rate = tf.get_variable("learning_rate", [], tf.float32, tf.constant_initializer(0.001), trainable = False)

		model = Homo_Model_243(batch_size, seq_len, my_hash, 512, 512, clust_size = clust_size, is_training = True)

		loss = model.loss(cat_batch, kl_strength)
		opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

	restorer = tf.train.Saver(tf.global_variables())
	saver = tf.train.Saver(tf.global_variables())

	summary_op = tf.summary.merge_all()
	sess = tf.Session()

	init = tf.global_variables_initializer()

	writer = tf.summary.FileWriter(log_dir + "/" + log_subdir + "/")
	writer.add_graph(tf.get_default_graph())
	run_metadata = tf.RunMetadata()

	tf_vars = tf.trainable_variables()
	total_params = 0
	for v in tf_vars:
		total_params += np.prod([int(xa) for xa in v.get_shape()])
	print("Total parameters: %i" %(total_params))

	my_hash.init.run(session = sess)

	threads = tf.train.start_queue_runners(sess = sess, coord = coord)
	reader.start_threads(sess)
	print("Started threads")

	if do_restore:
		restorer.restore(sess, file_to_restore)
	else:
		sess.run(init)
	print("Initialized variables.")

	begin = time.time()
	last_update = begin

	print("Finalizing graph.")
	tf.get_default_graph().finalize()
	steps_per_epoch = 1280000 // batch_size
	num_epochs = 20

	step = 0
	save_step = 2
	last_mean_aa = None

	try:
		for epoch in range(num_epochs):
			for iter in range(steps_per_epoch):
				fetches = [loss, opt, increase_kl]

				if step < 10 or step % 100 == 0:
					fetches += [summary_op]

				fetched = sess.run(fetches)

				mean_loss = fetched[0]
				if step < 10 or step % 100 == 0:
					print("Iteration %i, mean_loss is %.2f" %(step, mean_loss))
					writer.add_summary(fetched[-1], step)
					writer.flush()
				if step % 10000 == 0:
					saver.save(sess, log_dir + "/" + log_subdir + "/ckpt", global_step = step)
				step += 1
		print("Done training")
		saver.save(sess, log_dir + "/" + log_subdir + "/final_ckpt.ckpt")
		end_training(sess, coord, threads)

	except KeyboardInterrupt:
		print("Training interrupted, saving final checkpoint. Please wait a second.")
		saver.save(sess, log_dir + "/" + log_subdir + "/interrupt_%i_ckpt.ckpt" %(step))
		end_training(sess, coord, threads)
		print("Done saving checkpoint. Model should now quit.")

if __name__ == "__main__":
	log_dir, log_subdir = setup_log("test_vae_")
	homo_train(log_dir, log_subdir)

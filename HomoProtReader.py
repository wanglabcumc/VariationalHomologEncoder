import os, sys

"""Fasta files are expected to look like this ("..." to indicate amino acids omitted for space)...

>34e6ee6_34e6ee6
MAKE...KA*
>34e6ee6_f2648d
MAKE...RT*
>34e6ee6_f10e9a
MAKD...RI*
>34e6ee6_10b54d2
MIKD...KA*
>3646962_3646962
MFLK...DE*
>3646962_21cf7e3
MWIK...ED*

So each ID is composed of two parts... the representative, and the protein ID (I use hexadecimal counters for the protein/rep IDs).
So 34e6ee6_34e6ee6 says the protein ID is the same as representative (so it is the representative of the cluster).
But 34e6ee6_f2648d says the protein ID is f2648d and its representative is still same.
You'll note that the proteins have "*" at the end and only letters in amino acid alphabet are permitted.
"""

if sys.version_info > (3, 0):
	import queue as Q
else:
	import Queue as Q

import threading
import random

import tensorflow as tf
import numpy as np

def true_aa(seq):
	if "B" in seq or "J" in seq or "O" in seq or "U" in seq or "X" in seq or "Z" in seq:
		return False
	else:
		return True

def load_fasta(in_read, first_word=False, no_non_aa = False):
	seqs = {}
	cur_head = ""
	cur_seq = ""
	for line in in_read:
		if line[0] == ">":
			if cur_head != "":
				if not no_non_aa or true_aa(cur_seq):
					seqs[cur_head] = cur_seq
			cur_seq = ""
			if first_word:
				cur_head = line[1:].split(" ")[0].strip()
			else:
				cur_head = line[1:].strip()
		else:
			cur_seq += line.strip()
	if not no_non_aa or true_aa(cur_seq):
		seqs[cur_head] = cur_seq

	return seqs

def transform_to_array(prot_seq, max_prot_len):
	padded_seq = prot_seq.ljust(max_prot_len, "-")
	aa_to_int = {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
					'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '*': 21}
	numeric_seq = [aa_to_int[aa] for aa in padded_seq]
	return numeric_seq

def arrayify(numeric_seq):
	return np.asarray(numeric_seq).astype(np.int32).reshape(-1, 1)

def organize_all_proteins(file_name, max_prot_len, batch_size, set_length, fixed_rep = False):
	in_file = open(file_name)
	in_read = in_file.readlines()
	in_file.close()

	seqs = load_fasta(in_read, max_prot_len)

	#reps = {k.split("_")[0]: None for k in seqs.keys()}
	#for k in seqs.keys():
		#if k.split("_")[0] == k.split("_")[1]:
			#reps[k.split("_")[0]] = seqs[k]

	seq_clusts = {k.split("_")[0]: [] for k in seqs.keys()}
	for k in seqs.keys():
		k_id = k.split("_")[0]
		#if k.split("_")[0] != k.split("_")[1]:
		seq_clusts[k_id].append(seqs[k])

	sck = list(seq_clusts.keys())

	random.shuffle(sck)

	while True:
		for clust_count in range(len(sck)):
			clust = sck[clust_count]
			target_cluster = seq_clusts[clust]
			#random.shuffle(target_cluster)
			picks = target_cluster #[:] #set_length]
			#rand_rep = picks.pop(int(random.random() * set_length))
			#if fixed_rep:
				#pick_rep = reps[clust.split("_")[0]]
				#cur_batch = transform_to_array(pick_rep, max_prot_len)
				#for clust_member in picks[:(set_length - 1)]:
					#cur_batch += transform_to_array(clust_member, max_prot_len)
			#else:
			#picks = target_cluster #+ [reps[clust.split("_")[0]], ]
			random.shuffle(picks)
			cur_batch = []
			for clust_member in picks[:set_length]:
				cur_batch += transform_to_array(clust_member, max_prot_len)

			yield arrayify(cur_batch)
		random.shuffle(sck)

class HomoProtReader(object):
	def __init__(self, file_name, coord, batch_size, max_prot_len = 320, set_length = 4, queue_size = 10000):
		self.file_name = file_name
		self.coord = coord
		self.batch_size = batch_size
		self.max_prot_len = max_prot_len
		self.set_length = set_length
		self.threads = []
		self.cat_array = tf.placeholder(dtype = tf.int32, shape = None)
		#self.indiv_array = tf.placeholder(dtype = tf.int32, shape = None)

		#self.queue = tf.FIFOQueue(queue_size, ['int32', 'int32'], shapes=[(self.max_prot_len, 1), (self.max_prot_len, 1)])
		#self.enqueue = self.queue.enqueue([self.clust_array, self.indiv_array])

		self.queue = tf.FIFOQueue(queue_size, ['int32'], shapes=[(self.max_prot_len * set_length, 1)])
		self.enqueue = self.queue.enqueue([self.cat_array])

	def dequeue(self, num_elements):
		output = self.queue.dequeue_many(num_elements)
		return output

	def thread_main(self, sess):
		stop = False
		while not stop:
			iterator = organize_all_proteins(self.file_name, self.max_prot_len, self.batch_size, self.set_length)
			for cat_prot in iterator:
				if self.coord.should_stop():
					stop = True
					break
				sess.run(self.enqueue, feed_dict = {self.cat_array: cat_prot})

	def start_threads(self, sess, n_threads = 1):
		for _ in range(1):
			thread = threading.Thread(target=self.thread_main, args=(sess,))
			thread.daemon = True
			thread.start()
			self.threads.append(thread)
		return self.threads

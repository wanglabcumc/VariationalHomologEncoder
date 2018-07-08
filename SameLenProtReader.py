#We are just going to be much closer to WaveNet people here...
#SameLenProtReader is going to try to make queues that return proteins only of same length for each batch.
#This should save time on computation and seems reasonable to do.

import os
import sys

if sys.version_info > (3, 0):
	import queue as Q
else:
	import Queue as Q

import threading
import random

import tensorflow as tf
import numpy as np

def load_fasta(in_read, max_len=None):
	seqs = []
	cur_seq = ""
	for line in in_read:
		if line[0] == ">":
			if cur_seq != "":
				if max_len is None or len(cur_seq) < max_len:
					seqs.append(cur_seq)
			cur_seq = ""
		else:
			cur_seq += line.strip()
	if max_len is None or len(cur_seq) <= max_len:
		seqs.append(cur_seq)
	return seqs #so this is a list.

#We want to encode proteins as ndarrays that are like (prot_len, 1) in shape. We also pad them.
def transform_to_array(prot_seq, max_prot_len):
	padded_seq = prot_seq.ljust(max_prot_len, '-')
	aa_to_int = {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
					'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '*': 21}
	#print "14 aa"
	#aa_to_int = {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 4, 'L': 8, 'M': 8, 'N': 9, 'P': 10, 'Q': 11, 'R': 4, 'S': 11, 'T': 11, 'V': 8, 'W': 12, 'Y': 5, '*': 13}
	#aa_to_int = {'-': 0, 'A': 1, 'C': 2, 'D': 4, 'E': 4, 'F': 5, 'G': 1, 'H': 3, 'I': 1, 'K': 3, 'L': 1, 'M': 1, 'N': 2, 'P': 2, 'Q': 2, 'R': 3, 'S': 2, 'T': 2, 'V': 1, 'W': 5, 'Y': 5, '*': 6}
	numeric_seq = [aa_to_int[aa] for aa in padded_seq]
	array_seq = np.asarray(numeric_seq).astype(np.int32).reshape(-1, 1)
	return array_seq

def validate_all_proteins(file_name, max_prot_len, batch_size):
	in_file = open(file_name)
	in_read = in_file.readlines()
	in_file.close()

	seqs = load_fasta(in_read, max_prot_len)

	#for batch_length in batch_list:
		#for mb in range(batch_size):
	for mb in range(len(seqs) // batch_size):
		for batch_item in range(batch_size):
			yield transform_to_array(seqs[mb * batch_size + batch_item] + "*", max_prot_len) #maybe +1 for the stop codon I guess?
	print "Done with file."

def organize_all_proteins(file_name, max_prot_len, batch_size):
	print("Reading fasta file", file_name)
	in_file = open(file_name)
	in_read = in_file.readlines()
	in_file.close()

	seqs = load_fasta(in_read, max_prot_len)

	random.shuffle(seqs)

	#We divvy up the batches ahead of time.
	#We'll map out exactly what we have to iterate through.
	seq_lens = [len(sq) for sq in seqs]
	observed_lens = list(set(seq_lens))

	batch_list = []
	for sq_ln in observed_lens:
		num_prots_of_length = seq_lens.count(sq_ln)
		num_batches = num_prots_of_length // batch_size #Python 3 compatible.
		for nb in range(num_batches):
			batch_list.append(sq_ln)

	#batch_list now has the full set of seq_len indices to iterate over.
	#we can just iterate through these and should never hit an error.
	#hmm...
	#it just occurred to me with multiple queues we may see same proteins
	#several times... can we just keep it to one thread?
	#We'll see if this slows down performance. Hopefully it doesn't...
	#using only one thread makes a lot of stuff easier, I guess.

	#print "Constructed queues..."
	#print "\n\nWARNING, NO STOP CODON!!!!\n\n"
	#Okay now load them into length-dependent queues...
	while True:
		print("Refreshing queues...")
		random.shuffle(batch_list)
		random.shuffle(seqs)
		prot_queues = {}
		for length in observed_lens:
			prot_queues[length] = Q.Queue()

		for seq in seqs:
			seq_len = len(seq)
			prot_queues[seq_len].put(seq)

		#Now we sample lengths at random and only add batch_size
		#elements at a time.
		#The hope is that this results in a queue that has
		#fixed lengths for each batch.
		#One thing to watch out for is whether queues ever "cross-feed"
		#meaning that batches would mix up or get out of modulo.
		for batch_length in batch_list:
			for mb in range(batch_size):
				yield transform_to_array(prot_queues[batch_length].get() + "*", max_prot_len) #maybe +1 for the stop codon I guess?

	#Okay so now we need to maintain some arrays indexed by length...

def load_all_proteins(file_name, max_prot_len):
	#print "Reading fasta file", file_name
	in_file = open(file_name)
	in_read = in_file.readlines()
	in_file.close()

	seqs = load_fasta(in_read)

	while True:
		#Shuffle the sequences on each read-through of file.
		random.shuffle(seqs)
		for seq in seqs:
			if len(seq) < max_prot_len:
				yield transform_to_array(seq + "*", max_prot_len)

def load_generic_proteins(file_name, max_prot_len):
	#We need to make a generator that returns protein sequences...
	directory = "/".join(file_name.split("/")[:-1])
	file_prefix = file_name.split("/")[-1]
	list_of_files = []
	for f in os.listdir(directory):
		if f.startswith(file_prefix):
			list_of_files.append(directory + "/" + f)

	while True:
		specific_file = list_of_files[int(random.random() * len(list_of_files))]
		with open(specific_file) as in_file:
			in_read = in_file.readline()
			cur_head = ""
			cur_seq = ""
			while in_read:
				if in_read[0] == ">":
					if cur_seq != "" and len(cur_seq) < max_prot_len: #I want to get rid of proteins exactly 256.
						yield transform_to_array(cur_seq + "*", max_prot_len) #Important that generator learns to end proteins.
					cur_seq = ""
					cur_head = in_read[1:].strip()
				else:
					cur_seq += in_read.strip()
				in_read = in_file.readline()
			if cur_seq != "" and len(cur_seq) < max_prot_len:
				yield transform_to_array(cur_seq + "*", max_prot_len)
			in_file.close()

class NewProtReader(object):

	def __init__(self,
					file_name,
					coord,
					batch_size,
					max_prot_len = 256,
					queue_size=10000): #1024):
		self.file_name = file_name
		self.coord = coord
		self.batch_size = batch_size
		self.max_prot_len = max_prot_len
		self.threads = []
		self.sample_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
		#print queue_size
		self.queue = tf.FIFOQueue(queue_size, ['int32'], shapes=[(self.max_prot_len, 1)])
		self.enqueue = self.queue.enqueue([self.sample_placeholder])

	def dequeue(self, num_elements):
		output = self.queue.dequeue_many(num_elements)
		return output

	def thread_main(self, sess):
		stop = False
		while not stop:
			iterator = validate_all_proteins(self.file_name, self.max_prot_len, self.batch_size)
			print "VALIDATION MODE!"
			for prot_array in iterator:
				if self.coord.should_stop():
					stop = True
					break
				sess.run(self.enqueue, feed_dict={self.sample_placeholder: prot_array})

	def start_threads(self, sess, n_threads = 1):
		for _ in range(n_threads):
			thread = threading.Thread(target=self.thread_main, args=(sess,))
			thread.daemon = True
			thread.start()
			self.threads.append(thread)
		return self.threads

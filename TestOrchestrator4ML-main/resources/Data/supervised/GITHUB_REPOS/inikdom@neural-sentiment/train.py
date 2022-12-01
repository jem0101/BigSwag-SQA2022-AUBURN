'''
I used mainly the tensorflow translation example:
https://github.com/tensorflow/tensorflow/

and semi-based this off the sentiment analyzer here:
http://deeplearning.net/tutorial/lstm.html

Written by: Dominik Kaukinen
'''
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import sys
import math
import os
try:
	import ConfigParser
except ImportError:
	import configparser as ConfigParser
import random
import time
from six.moves import xrange
import util.dataprocessor
import util.hyperparams as hyperparams
import models.sentiment
import util.vocabmapping

#Defaults for network parameters

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "config.ini", "Path to configuration file with hyper-parameters.")
flags.DEFINE_string("data_dir", "data/", "Path to main data directory.")
flags.DEFINE_string("checkpoint_dir", "data/checkpoints/", "Directory to store/restore checkpoints")

def main():
	checkpoint_dir, hyper_params = get_ckpt_path_params()
	util.dataprocessor.run(hyper_params["max_seq_length"],
		hyper_params["max_vocab_size"])

	#create model
	print("Creating model with...")
	print("Number of hidden layers: {0}".format(hyper_params["num_layers"]))
	print("Number of units per layer: {0}".format(hyper_params["hidden_size"]))
	print("Dropout: {0}".format(hyper_params["dropout"]))
	vocabmapping = util.vocabmapping.VocabMapping()
	hyper_params["max_vocab_size"] = vocabmapping.get_size()
	print("Vocab size is: {0}".format(hyper_params["max_vocab_size"]))
	path = os.path.join(FLAGS.data_dir, "processed/")
	infile = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	data = np.load(os.path.join(path, infile[0]))
	for i in range(1, len(infile)):
		data = np.vstack((data, np.load(os.path.join(path, infile[i]))))
	np.random.shuffle(data)

	num_batches = int(len(data) / hyper_params["batch_size"])
	# split for train/test
	train_start_end_index = [0, int(hyper_params["train_frac"] * len(data))]
	test_start_end_index = [int(hyper_params["train_frac"] * len(data)) + 1, len(data) - 1]
	print("Number of training examples per batch: {0}, \
	\nNumber of batches per epoch: {1}".format(hyper_params["batch_size"],num_batches))
	with tf.Session() as sess:
		writer = tf.summary.FileWriter("/tmp/tb_logs", sess.graph)
		model = create_model(sess, hyper_params, checkpoint_dir)
	#train model and save to checkpoint
		print("Beggining training...")
		print("Maximum number of epochs to train for: {0}".format(hyper_params["max_epoch"]))
		print("Batch size: {0}".format(hyper_params["batch_size"]))
		print("Starting learning rate: {0}".format(hyper_params["learning_rate"]))
		print("Learning rate decay factor: {0}".format(hyper_params["lr_decay_factor"]))

		step_time, loss = 0.0, 0.0
		previous_losses = []
		tot_steps = int(num_batches * hyper_params["max_epoch"])
		model.init_data(data, train_start_end_index, test_start_end_index)
		#starting at step 1 to prevent test set from running after first batch
		for step in xrange(1, tot_steps):
			# Get a batch and make a step.
			start_time = time.time()

			inputs, targets, seq_lengths = model.get_batch()
			str_summary, step_loss, _ = model.step(sess, inputs, targets, seq_lengths, False)

			step_time += (time.time() - start_time) / hyper_params["steps_per_checkpoint"]
			loss += step_loss / hyper_params["steps_per_checkpoint"]

			# Once in a while, we save checkpoint, print statistics, and run evals.
			if step % hyper_params["steps_per_checkpoint"] == 0:
				writer.add_summary(str_summary, step)
				# Print statistics for the previous epoch.
				print("global step %d learning rate %.7f step-time %.2f loss %.4f"
				% (model.global_step.eval(), model.learning_rate.eval(),
				step_time, loss))
				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)
				# Save checkpoint and zero timer and loss.
				step_time, loss, test_accuracy = 0.0, 0.0, 0.0
				# Run evals on test set and print their accuracy.
				print("Running test set")
				for test_step in xrange(len(model.test_data)):
					inputs, targets, seq_lengths = model.get_batch(True)
					str_summary, test_loss, _, accuracy = model.step(sess, inputs, targets, seq_lengths, True)
					loss += test_loss
					test_accuracy += accuracy
				normalized_test_loss, normalized_test_accuracy = loss / len(model.test_data), test_accuracy / len(model.test_data)
				checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "sentiment{0}.ckpt".format(normalized_test_accuracy))
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				writer.add_summary(str_summary, step)
				print("Avg Test Loss: {0}, Avg Test Accuracy: {1}".format(normalized_test_loss, normalized_test_accuracy))
				print("-------Step {0}/{1}------".format(step,tot_steps))
				loss = 0.0
				sys.stdout.flush()

def create_model(session, hyper_params, path):
	model = models.sentiment.SentimentModel(vocab_size = hyper_params["max_vocab_size"],
											hidden_size = hyper_params["hidden_size"],
											dropout = hyper_params["dropout"],
											num_layers = hyper_params["num_layers"],
											max_gradient_norm = hyper_params["grad_clip"],
											max_seq_length = hyper_params["max_seq_length"],
											learning_rate = hyper_params["learning_rate"],
											lr_decay = hyper_params["lr_decay_factor"],
											batch_size = hyper_params["batch_size"])
	ckpt_path = tf.train.latest_checkpoint(path)
	if ckpt_path:
		print("Reading model parameters from {0}".format(ckpt_path))
		model.saver.restore(session,ckpt_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
	return model

def read_config_file():
	'''
	Reads in config file, returns dictionary of network params
	'''
	config = ConfigParser.ConfigParser()
	config.read(FLAGS.config_file)
	dic = {}
	sentiment_section = "sentiment_network_params"
	general_section = "general"
	dic["num_layers"] = config.getint(sentiment_section, "num_layers")
	dic["hidden_size"] = config.getint(sentiment_section, "hidden_size")
	dic["dropout"] = config.getfloat(sentiment_section, "dropout")
	dic["batch_size"] = config.getint(sentiment_section, "batch_size")
	dic["train_frac"] = config.getfloat(sentiment_section, "train_frac")
	dic["learning_rate"] = config.getfloat(sentiment_section, "learning_rate")
	dic["lr_decay_factor"] = config.getfloat(sentiment_section, "lr_decay_factor")
	dic["grad_clip"] = config.getint(sentiment_section, "grad_clip")
	dic["use_config_file_if_checkpoint_exists"] = config.getboolean(general_section,
		"use_config_file_if_checkpoint_exists")
	dic["max_epoch"] = config.getint(sentiment_section, "max_epoch")
	dic ["max_vocab_size"] = config.getint(sentiment_section, "max_vocab_size")
	dic["max_seq_length"] = config.getint(general_section,
		"max_seq_length")
	dic["steps_per_checkpoint"] = config.getint(general_section,
		"steps_per_checkpoint")
	return dic

def get_ckpt_path_params():
	'''
	Retrieves hyper parameter information from either config file or checkpoint
	'''
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	serializer = hyperparams.HyperParameterHandler(FLAGS.checkpoint_dir)
	hyper_params = read_config_file()
	checkpoint_dir = "maxseqlen_{0}_hidden_size_{1}_numlayers_{2}_vocab_size_{3}".format(
	hyper_params["max_seq_length"],
	hyper_params["hidden_size"],
	hyper_params["num_layers"],
	hyper_params["dropout"])
	new_checkpoint_dir = os.path.join(FLAGS.checkpoint_dir,
		checkpoint_dir)
	if os.path.exists(new_checkpoint_dir):
		print("Existing checkpoint found, loading...")
	else:
		os.makedirs(new_checkpoint_dir)
		serializer = hyperparams.HyperParameterHandler(new_checkpoint_dir)
		serializer.save_params(hyper_params)
	return new_checkpoint_dir, hyper_params

if __name__ == '__main__':
	main()

import sys
import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, SimpleRNN, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow.keras.utils as utils
import tensorflow.keras.backend as K
import pickle

# python predictive_RNN.py [train/test] [model] [layer_size] [test_slice]

# Constants for all models.
INPUT_FILE = 'cumlen20_nomod.txt' # 512*170
CORPUS = 'LEQ_20_NOMOD'
TEST_INPUT = './eval_data/ND_leq_20.txt' # 512*17
EXPERIMENT = 'ND'
SEQ_LENGTH = 1
EPOCHS = 50
BATCH = 512
VALIDATION = 0.00 # Running training without validation.
MODE = sys.argv[1]
NETWORK = sys.argv[2]
LAYER_SIZE = int(sys.argv[3])
if MODE == 'test':
	TEST_SLICE = int(sys.argv[4])
else:
	TEST_SLICE = 0

# Various callbacks for later model training/evaluation.
# Ran into an odd bug where tf could not save files when the path was specified with '\dir\dir\', using '\\' instead of '\' works.

# Directory where the checkpoints will be saved.
checkpoint_dir = '.\\saved_models\\' + CORPUS + '\\' + str(NETWORK) + '\\' + str(LAYER_SIZE) + '\\'
# Name of the checkpoint files.
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch:02d}-{loss:.4f}-{ignore_acc:.4f}-{categorical_accuracy:.4f}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
#stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)
# Define the Keras TensorBoard callback.
logdir = '.\\logs\\fit\\' + CORPUS + '\\' + str(NETWORK) + '\\' + str(LAYER_SIZE) + '\\'
tensorboard = TensorBoard(log_dir=logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Preparing training/validation data.
file = open(INPUT_FILE, encoding = 'utf8')
raw_text = file.read()
chars = sorted(list(set(raw_text)))
print(chars) # Sanity check 1.

text_length = len(raw_text)
char_length = len(chars)
VOCABULARY = char_length
print("Text length = " + str(text_length)) # Sanity check 2.
print("No. of characters = " + str(char_length)) # Sanity check 3.
# Translating characters to integers to input into model and vice-versa to interpret model output.
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = np.array(chars)
print(char_to_int) # Sanity check 4.
input_strings = []
output_strings = []
file.close()

def ignore_acc(y_true, y_pred):
	"""Custom accuracy calculation, based on the reasoning in Bernardy (2018). Only evaluates predictions on closing brackets.
	"""
	# Find class index.
	y_true_class = K.argmax(y_true, axis=-1)
	y_pred_class = K.argmax(y_pred, axis=-1)

	# Mask both opening brackets by comparing found class index to character indices.
	ignore_square = K.cast(K.not_equal(y_pred_class, char_to_int['[']), 'int32')
	ignore_curl = K.cast(K.not_equal(y_pred_class, char_to_int['{']), 'int32')
	ignore_EOS = K.cast(K.not_equal(y_pred_class, char_to_int['$']), 'int32')
	ignore_mask = ignore_square * ignore_curl * ignore_EOS

	# Only evaluate accuracy on unmasked characters.
	matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
	accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
	return accuracy

def buildmodel(vocabulary, layer_size, network):
	"""Builds a neural network model with one hidden layer.
	Args:
		vocabulary: Size of the vocabulary the model is trained upon. Determines number of neurons in the Dense output layer.
		layer_size: Number of neurons in the hidden layer.
		network: Determines the type of the hidden layer. Either LSTM, GRU or Simple RNN.
		
	Returns:
		model - Keras model with one hidden layer of the specified parameters.
	"""
	
	model = Sequential()
	if network == 'LSTM':	
		model.add(LSTM(layer_size, input_shape = (SEQ_LENGTH, 1), return_sequences = False))
	elif network == 'GRU':
		model.add(GRU(layer_size, input_shape = (SEQ_LENGTH, 1), return_sequences = False))
	elif network == 'SRNN':
		model.add(SimpleRNN(layer_size, input_shape = (SEQ_LENGTH, 1), return_sequences = False))	
	else:
		"Could not build model. Provide a valid network argument in the command line.\n\tpython predictive_RNN.py [train/test] [LSTM/SRNN/GRU]"
		return 0
	model.add(Dense(vocabulary, activation = 'softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = tf.optimizers.Adam(learning_rate=0.0001), metrics = [ignore_acc, 'categorical_accuracy'])
	model.summary()
	return model

def evaluate(word, model, vocabulary):
	"""Evaluate model accuracy according to Bernardy (2018).
	Prediction of the model is evaluated for each index in the input word if and only if word[i] is a closing bracket.
	
	Args:
		word: Input word, string.
		model: Model to be evaluated, tf.keras.model.
		vocabulary: Size of the training corpus vocabulary.
		
	Returns:
		accuracy_by_index: List of the accuracy measurements per character.
		predictions_by_index: List of the full output of the RNN per character.
	"""
	correct_preds = 0
	closing_brackets = 0
	accuracy_by_index = []
	predictions_by_index = []
	for i in range(len(word)-1):
		X = np.reshape(word[i], (1, SEQ_LENGTH, 1))
		Y_pred = model.predict(X/float(vocabulary))
		predictions_by_index.append(Y_pred)
		index = np.argmax(Y_pred)
		if word[i+1] == char_to_int['}'] or word[i+1] == char_to_int[']']:
			closing_brackets += 1
			if index == word[i+1]:
				correct_preds += 1
		if closing_brackets:
			accuracy = correct_preds/float(closing_brackets)
		else:
			# Accuracy via definitionem can't be lower than 1.0 before encountering a closing bracket.
			accuracy = 1.
		accuracy_by_index.append(accuracy)
	return accuracy_by_index, predictions_by_index

def corpus_eval(words, splits):
	"""Calculates accuracy on the chunked test corpus.
		Args:
			words: Preprocessed test corpus.
			splits: Maximum number of splits.
		Returns:
			avg_accuracy_by_index: List containing accuracy by index, averaged over every word in the corpus.
			accuracy_by_index: List containing seperate accuracy by index for every word in the corpus.
	"""
	# Evaluation is done by splitting the testing corpus into 169 pieces with 51 words each + a final piece for all leftover words, evaluating on those and then taking the average from all the shorter sets.
	# This is because otherwise, repeatedly calling model.predict() caused
	# a memory leak on my build.
	# https://github.com/keras-team/keras/issues/13118
	# K.clear_session() + reloading the model for
	# a continuous evaluation resulted in a slower, but still significant leak.
	pickle_dir = './pickle/' + str(NETWORK) + '/' + str(LAYER_SIZE) + '/' + CORPUS + '/' + EXPERIMENT + '/'
	size = len(words)
	chunk = int(size/splits)

	corpus_accuracy_by_index = []
	c = 0 # Solely for output.
	# Set start and end of current chunk being evaluated.
	if TEST_SLICE == 170:
		start = 51*170
		end = len(words)
	else:
		start = TEST_SLICE * chunk
		end = start + chunk
	predictions_by_word = []
	print(" ======= SPLIT {} ======= ".format(TEST_SLICE))
	for word in words[start:end]:
		c += 1
		result, predictions = evaluate(word, model, VOCABULARY)
		corpus_accuracy_by_index.append(result)
		predictions_by_word.append(predictions)
		if c % 10 == 0:
			print("{}/{}".format(c, chunk))
	# Save chunk results.
	pickle.dump(corpus_accuracy_by_index, open(pickle_dir + "corpus_accuracy_p" + str(TEST_SLICE) + ".p", "wb"))
	pickle.dump(predictions_by_word, open(pickle_dir + "predictions_by_word_p" + str(TEST_SLICE) + ".p", "wb"))

	corpus_accuracy_by_index = []
	for i in range(splits+1):
		corpus_accuracy = pickle.load(open(pickle_dir + "corpus_accuracy_p" + str(i) + ".p", "rb"))
		corpus_accuracy_by_index = corpus_accuracy_by_index + corpus_accuracy
	# Accuracy by index over the entire corpus is average of accuracy at each index through the corpus.
	avg_accuracy_by_index = [float(sum(i))/size for i in zip(*corpus_accuracy_by_index)]
	return [avg_accuracy_by_index, corpus_accuracy_by_index]
	

if MODE == 'train':
	# Prepare input (X) and target (Y) data.
	for i in range(len(raw_text) - SEQ_LENGTH):
		X_text = raw_text[i:i + SEQ_LENGTH]
		X = [char_to_int[char] for char in X_text]
		input_strings.append(X)	
		Y = raw_text[i + SEQ_LENGTH]
		output_strings.append(char_to_int[Y])
	
	input_strings = np.array(input_strings)
	input_strings = np.reshape(input_strings, (input_strings.shape[0], input_strings.shape[1], 1))
	input_strings = input_strings/float(VOCABULARY) # active for main training

	output_strings = np.array(output_strings)
	output_strings = utils.to_categorical(output_strings)
	print(input_strings.shape)
	print(output_strings.shape)

	# Create and train the model.
	model = buildmodel(VOCABULARY, LAYER_SIZE, NETWORK)
	history = model.fit(input_strings, output_strings, validation_split = VALIDATION, epochs = EPOCHS, batch_size = BATCH, callbacks = [checkpoint_callback])
elif MODE == 'test':
	# Prepare corpus to evaluate on.
	words = []
	infile = open(TEST_INPUT, 'r')
	for line in infile:
		raw_words = line.split('$')
		for raw_word in raw_words:
			raw_word = raw_word + '$'
			words.append([char_to_int[char]/float(VOCABULARY) for char in raw_word])
	words = words[:-1]
	evaluated_words = len(words)
	print("Words in test corpus: {}".format(evaluated_words))
	
	# Build the model based off of the latest checkpoint.

	model = buildmodel(VOCABULARY, LAYER_SIZE, NETWORK)
	model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
	model.compile(loss = 'categorical_crossentropy', metrics = [ignore_acc, 'categorical_accuracy'])
	model.summary()
	
	avg_accuracy_by_index, accuracy_by_index = corpus_eval(words, 170)
	
	if TEST_SLICE == 170: # Final slice of test corpus processed.
		out_avg = './eval/' + str(NETWORK) + '/' + str(LAYER_SIZE) + '/' + CORPUS + '/' + EXPERIMENT + '/' + 'avg_acc_by_index.p'
		pickle.dump(avg_accuracy_by_index, open(out_avg, "wb"))
		out_acc = './eval/' + str(NETWORK) + '/' + str(LAYER_SIZE) + '/' + CORPUS + '/' + EXPERIMENT + '/' + 'acc_by_index.p'
		pickle.dump(accuracy_by_index, open(out_acc, "wb"))
		print(avg_accuracy_by_index)
	
else:
	print("Please specify a mode - train or test.")
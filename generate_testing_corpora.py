# clean megaset

import random
import numpy as np

MAX_LENGTH = 20
TEST_LENGTH = MAX_LENGTH - 2
infile = open('cumlen20_nomod.txt','r')
words = set()
words_test1 = set()
words_test2 = set()
words_LRD = set() # LRD = long range dependency
words_ND = set() # ND = new depths

for line in infile:
	#print(line)
	raw_words = line.split('$')
	for word in raw_words:
		if len(word) == TEST_LENGTH:
			words_test1.add(word)
		if len(word) == MAX_LENGTH:
			words_test2.add(word)
		words.add(word)
infile.close()	

print("Test length {}".format(TEST_LENGTH))
print(len(words), len(words_test1), len(words_test2))

# Create set for long-range dependency experiment.
# Concatenate two TEST_LENGTH substrings and wrap them in
# a random pair of brackets.
print("Creating LRD set ....")
for i in range(len(words_test1)):
	random_float = np.random.uniform(0.0, 1.0)
	w1 = random.sample(words_test1, 1)[0]
	w2 = random.sample(words_test1, 1)[0]
	if random_float < 0.5:
		new_word = '[' + w1 + w2 + ']'
	else:
		new_word = '{' + w1 + w2 + '}'
	words_LRD.add(new_word)
	
# Create set for new depths experiment.
# Take a MAX_LENGTH string and wrap it in five
# pairs of brackets
print("Creating ND set ....")
for i in range(len(words_test2)):
	new_word = random.sample(words_test2, 1)[0]
	for i in range(5):
		random_float = np.random.uniform(0.0, 1.0)
		if random_float < 0.5:
			new_word = '[' + new_word + ']'
		else:
			new_word = '{' + new_word + '}'
	words_ND.add(new_word)

print("Len LRD: {}".format(len(words_LRD)))
print("Len ND: {}".format(len(words_ND)))
print("Saving eq ....")
outfile = open('./eval_data/train_eq_' + str(MAX_LENGTH) + '.txt','w')
for word in words_test2:
	outfile.write(word)
outfile.close()

print("Saving LRD ....")
words_LRD = list(words_LRD)
outfile = open('./eval_data/LRD_leq_' + str(MAX_LENGTH) + '.txt','w')
for word in words_LRD[:512*17]:
	outfile.write("{}{}".format(word,'$'))
outfile.close()

print("Saving ND ....")
words_ND = list(words_ND)
outfile = open('./eval_data/ND_leq_' + str(MAX_LENGTH) + '.txt','w')
for word in words_ND[:512*17]:
	outfile.write("{}{}".format(word,'$'))
outfile.close()
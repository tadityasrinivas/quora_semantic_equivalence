'''
This is useful for creating a 'word->glove-vec' dictionary with only necessary words
that occur in the Quora dataset. This makes it possible to load the
dictionary of only necessary GloVe vectors into memory, without having to 
keep the word vectors of all 1.9 million words in memory.
'''
import numpy as np
import cPickle
import random

# Load all the data examples
# Structure: (sentence 1, sentence 2, 0/1 label)
with open("data/data_tuples_glovem.p", "rb") as f:
	data_tuples = cPickle.load(f)

needed_glove_words = {} # Creating this dict for O(1) look-up
for tup in data_tuples:
	s1 = tup[0]
	s2 = tup[1]
	for word in s1.split():
		if word not in needed_glove_words:
			needed_glove_words[word] = 1 # Some lightweight dummy value

	for word in s2.split():
		if word not in needed_glove_words:
			needed_glove_words[word] = 1
	
print "Num of needed glove words ", len(needed_glove_words)

# Create the needed glove vector dictionary
# This assumes that the glove vector file has been downloaded from the link in readme
vec_file = open('data/glove_raw/glove.42B.300d.txt')
needed_glove_dict = {}
num_words = 0
for line in vec_file:
	words = line.split()
	if words[0] in needed_glove_words:
		needed_glove_dict[words[0]] = np.asarray(words[1:])

	num_words += 1
	if num_words%100000 == 0:
		print "Processed so far: ", num_words/100000, " x100K"

vec_file.close()
print "Done creating Glove vectors dict for necessary words"
print "Number of words ", len(needed_glove_dict)

with open("data/needed_glovem_dict.p", "wb") as f:
	cPickle.dump(needed_glove_dict, f)

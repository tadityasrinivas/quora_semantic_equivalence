import numpy as np
import csv
import cPickle
import re
import random
from spell import correction
import sys

def process_word(in_word):
	# All processing has been shifted to 'process_sentence'. Refer to its implementation.
	return in_word.lower()

def process_sentence(in_sent):
	in_sent = in_sent.strip(" ?")
	in_sent = in_sent.replace("/", " ")
	in_sent = in_sent.replace("-", " ")
	in_sent = in_sent.replace("(", " ")
	in_sent = in_sent.replace(")", " ")
	in_sent = in_sent.replace(".", " ")
	in_sent = in_sent.replace("?", " ")
	in_sent = in_sent.replace(",", " ")
	in_sent = in_sent.replace(";", " ")
	in_sent.strip()
	in_sent = re.sub(r'[^a-zA-Z0-9 ]','', in_sent)

	# Wherever a transition from num to alpha occurs, add space. Useful for stuff like: 25kg, Rs100 etc.
	out_sent_chars = [] #list of chars to output
	for ix, ch in enumerate(in_sent):
		if ix==0:
			out_sent_chars.append(in_sent[0])
			continue
		if (in_sent[ix-1].isdigit() and in_sent[ix].isalpha()) or (in_sent[ix-1].isalpha() and in_sent[ix].isdigit()):
			out_sent_chars.append(" ")
			out_sent_chars.append(in_sent[ix])
		else:
			out_sent_chars.append(in_sent[ix])
	out_sent = "".join(out_sent_chars) # Put them into a string
	return out_sent

# The following two lines assume that the pickle file contains a dictionary (set also works) of the form: glove word -> (any dummy value)
# The dictionary will be used later, to check in O(1) if a given word belongs to the list of 1.9M Glove words.
with open("data/all_glove1m_words.p", "rb") as f:
	all_glove_words = cPickle.load(f)

# Reading the Quora data
rows = []
with open('data/quora_duplicate_questions.tsv','rb') as tsvin:
	tsvin = csv.reader(tsvin, delimiter='\t')
	for idx, row in enumerate(tsvin):
		if idx > 0: # First row is fields data
			rows.append(row)

# Get all the words in the dataset
all_words_dict = {}
recog_cnt = 0
bad_words = {}
data_tuples = [] # List of tuples (s1, s2, label). Store processed sents
total_num_bad_tokens = 0
num_bad_sentence_pairs = 0
total_num_tokens = 0

for idx, row in enumerate(rows):
	if (idx + 1) % 1000 == 0:
		print "About to process example ", (idx+1)
		sys.stdout.flush()

	# Preprocess the sentences
	sent_1 = process_sentence(row[3])
	words_1 = sent_1.split()
	final_words_1 = [] # Words after all processing steps

	sent_2 = process_sentence(row[4])
	words_2 = sent_2.split()
	final_words_2 = [] # After all processing steps
	curr_sent_1_good = True
	curr_sent_2_good = True

	for word in words_1:
		total_num_tokens += 1
		word_root = process_word(word)
		if word_root not in all_words_dict:
			# Count the number of words in dataset that are in glove
			if word_root in all_glove_words:
				all_words_dict[word_root] = 1
				recog_cnt += 1
				final_words_1.append(word_root)
				continue # Since we're done with curr word

			# Attempt to spell correct, only if original word not in Glove words
			attempted_correction = correction(word_root)
			if attempted_correction in all_glove_words:
				final_words_1.append(attempted_correction)
				if attempted_correction not in all_words_dict:
					all_words_dict[attempted_correction] = 1
					recog_cnt += 1
			else:
				all_words_dict[word_root] = 1
				final_words_1.append(word_root)
				bad_words[word_root] = 1
				curr_sent_1_good = False
				total_num_bad_tokens += 1
		else: # Word is already in all_words_dict
			final_words_1.append(word_root)
			if word_root in bad_words:
				curr_sent_1_good = False
				total_num_bad_tokens += 1
			continue
		
	# Current implementation tries to spell correct every repeated occurence of misspelt
	# words (since all_words_dict only stores corrected forms). instead, we could
	# maintain a cached dictionary from misspelt words so far to their corrections
	# Unrecognized words dont have this issue, since they are directly added to the dict
	for word in words_2:
		total_num_tokens += 1
		word_root = process_word(word)
		if word_root not in all_words_dict:
			# Count the number of words in dataset that are in glove
			if word_root in all_glove_words:
				all_words_dict[word_root] = 1
				recog_cnt += 1
				final_words_2.append(word_root)
				continue # Since we've fin curr word

			# Attempt to spell correct, only if original word not in Glove words
			attempted_correction = correction(word_root)
			if attempted_correction in all_glove_words:
				final_words_2.append(attempted_correction)
				if attempted_correction not in all_words_dict:
					all_words_dict[attempted_correction] = 1
					recog_cnt += 1
			else:
				all_words_dict[word_root] = 1
				final_words_2.append(word_root)
				bad_words[word_root] = 1
				curr_sent_2_good = False
				total_num_bad_tokens += 1
		else:
			final_words_2.append(word_root)
			if word_root in bad_words:
				curr_sent_2_good = False
				total_num_bad_tokens += 1
			continue

	if (not curr_sent_1_good) or (not curr_sent_2_good):
		num_bad_sentence_pairs += 1
	processed_sent_1 = " ".join(final_words_1)
	processed_sent_2 = " ".join(final_words_2)
	curr_label = int(row[5])
	data_tuples.append( (processed_sent_1, processed_sent_2, curr_label) )

print "Number of data_tuples collected ", len(data_tuples)

# Write tuples to disk
with open("data/data_tuples_glovem.p", "wb") as f:
	cPickle.dump(data_tuples,f)

print "Some remaining bad tokens. This should mostly be proper nouns at this point, and very badly misspelt words."
print random.sample(bad_words.keys(), 500)
print "SOME MORE USEFUL STATISTICS THAT HAVE BEEN TRACKED --------------"
print "Total number of words in data ", len(all_words_dict)
print "Number of words in data that are in Glove ", recog_cnt
print "Number of bad words ", len(bad_words)
print "Bad sentence pairs number ", num_bad_sentence_pairs
print "Total num of bad tokens in text ", total_num_bad_tokens
print "Total num of tokens in text ", total_num_tokens

# Handle Apostrophy s as a special case TODO
# Spell correction is a must - Used a simple one for now. Can try to use a more advanced one.
# Remove illegible characters (TM, French etc.) before any other processing TODO. Done as of 1/22.

#print len(rows)
#print rows[0]
#print rows[1]
#print rows[24]
#print rows[-1]

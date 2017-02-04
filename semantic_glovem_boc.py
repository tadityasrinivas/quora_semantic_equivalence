import numpy as np
from sklearn.svm import LinearSVC, SVC
import cPickle
import random
from string import ascii_lowercase as ascii_l

# Create a char-to-idx dictionary
all_chars = []
for ch in ascii_l:
	all_chars.append(ch)

for idx in range(10):
	all_chars.append(str(idx))
		
char_to_idx = {ch:ix for ix, ch in enumerate(all_chars)}
print len(char_to_idx)

# Load all the data examples
# (s1, s2, 0/1)
with open("data/data_tuples_glovem.p", "rb") as f:
	data_tuples = cPickle.load(f)

with open("data/needed_glovem_dict.p", "rb") as f:
	glove_dict = cPickle.load(f)

print "Done loading the necessary Glove vectors dict"

# Convert the data_tuples list to feature matrix & labels
num_bad = 23 # FROM ANALYZE_TUPLES, remove hardcoding
num_examples = len(data_tuples) - num_bad
glove_dim = 300
boc_dim = 36
feat_dim = glove_dim*2 +boc_dim*2 # GLOVE DIMENSION + BOC DIM
feat_matrix = np.zeros((num_examples, feat_dim))
labels = np.zeros(num_examples)

curr_idx = 0
for idx, curr_tuple in enumerate(data_tuples):
	if (idx + 1)%100000 == 0:
		print "Processing example ", idx+1

	sent1 = curr_tuple[0]
	sent2 = curr_tuple[1]

	if len(sent1.split()) == 0 or len(sent2.split())==0:
		continue

	boc1 = np.zeros(36)
	vec1 = np.zeros(glove_dim)
	denom = 0
	for word in sent1.split():
		if word in glove_dict:
			curr_vec = glove_dict[word]
			denom += 1
		else:
			for ch in word:
				boc1[char_to_idx[ch]] += 1
			continue
			#curr_vec = glove_dict['unk'] # Can choose how to handle unknown words
			#denom += 1
		vec1 += np.array(curr_vec, dtype=np.float)
	if denom != 0:
		vec1 /= denom # Averaging

	boc2 = np.zeros(36)
	vec2 = np.zeros(glove_dim)
	denom = 0
	for word in sent2.split():
		if word in glove_dict:
			curr_vec = glove_dict[word]
			denom += 1
		else:
			for ch in word:
				boc2[char_to_idx[ch]] += 1
			continue
			#curr_vec = glove_dict['unk'] # Choose how to handle unknown words
			#denom += 1
		vec2 += np.array(curr_vec, dtype=np.float)
	if denom != 0:
		vec2 /= denom # Averaging

	feat_matrix[curr_idx] = np.hstack((vec1, vec2, boc1, boc2)) # Why this order
	labels[curr_idx] = curr_tuple[2]
	curr_idx += 1

print "Done converting tuples to feat matrix"
print "Feat matrix size ", feat_matrix.shape
print "Last idx stored ", curr_idx-1

# Make the train/test split 80-20
train_pc = 0.8
test_pc = 1 - train_pc

idxes = np.arange(num_examples)
random.seed(186)
random.shuffle(idxes)

# Training data
train_matrix = feat_matrix[idxes[0: int(np.ceil(train_pc*num_examples)) ] ]
train_labels = labels[idxes[0: int(np.ceil(train_pc*num_examples)) ] ]

# Val data
val_matrix = feat_matrix[idxes[ int(np.ceil(train_pc*num_examples)) : ] ]
val_labels = labels[idxes[ int(np.ceil(train_pc*num_examples)) : ] ]

clf = LinearSVC(C=10.0, verbose=1)
#clf = SVC(C=10.0, verbose=1, kernel='poly', degree=2)
print "About to train SVM"
clf.fit(train_matrix, train_labels)
print "Done training"

val_pred = clf.predict(val_matrix)
acc = np.mean(val_labels == val_pred)

print "Validation Accuracy is ", acc
print "Training Accuracy is ", np.mean(train_labels == clf.predict(train_matrix) )


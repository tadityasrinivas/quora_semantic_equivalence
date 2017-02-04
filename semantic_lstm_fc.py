import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, merge, Input 
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import cPickle

# Create data matrices and labels list from processed data tuples
def create_data_matrices(in_data_tuples):
	X_1 = []
	X_2 = []
	labels = []
	for tup in in_data_tuples:
		s1 = tup[0]
		s2 = tup[1]
		label = tup[2]

		curr_seq_1 = []
		for word in s1.split():
			curr_seq_1.append(word_to_id[word])
			if len(curr_seq_1) == max_sentence_len: # Change this to pick best 60 (instead of 1st 60)
				break

		while True: # Keep padding 0's until max_sentence_len is reached
			if len(curr_seq_1) == max_sentence_len:
				break
			curr_seq_1.append(0)

		curr_seq_2 = []
		for word in s2.split():
			curr_seq_2.append(word_to_id[word])
			if len(curr_seq_2) == max_sentence_len:
				break

		while True: # Keep padding 0's until max_sentence_len is reached
			if len(curr_seq_2) == max_sentence_len:
				break
			curr_seq_2.append(0)

		X_1.append(np.array(curr_seq_1))
		X_2.append(np.array(curr_seq_2))
		labels.append(label)

	return np.array(X_1), np.array(X_2), labels

# load the processed Quora dataset
with open("data/data_tuples_glovem.p", "rb") as f:
	pre_data_tuples = cPickle.load(f)
print "Loaded the data tuples"

data_tuples = []
for tup in pre_data_tuples:
	if len(tup[0].split())==0 or len(tup[1].split())==0:
		continue
	data_tuples.append(tup)
print "Removed pairs with empty sentences. Remaining num. of data tuples ", len(data_tuples)

# Load glove vector dict (only for the needed words)
with open("data/needed_glovem_dict.p", "rb") as f:
	glove_dict = cPickle.load(f)

print "Loaded the Glove dictionary for necessary words"

glove_dim = glove_dict['the'].shape[0]
total_num_words = 80419 # Pass this from analyze_data, instead of hardcoding.

# Initialize embedding matrix with each entry sampled uniformly at random between -1.0 and 1.0
init_glove_matrix =  np.random.uniform(-1.0, 1.0, size=(total_num_words+1, glove_dim))
print "Initialized glove matrix with uniform. Will overwrite known vectors in it now"

# First create a dictionary from word to idx (for all distinct words)
word_to_id = {}
max_sentence_len = 0
sentence_lengths = []
curr_id = 1 # Start with 1, since 0 is used for <none> token (i.e., padding sentences to get to max length)
words_in_order = []
for tup in data_tuples:
	s1 = tup[0]
	s2 = tup[1]

	# Update max_sentence_len as necessary
	if len(s1.split()) > max_sentence_len:
		max_sentence_len = len(s1.split())
	if len(s2.split()) > max_sentence_len:
		max_sentence_len = len(s2.split())

	sentence_lengths.append(len(s1.split()))
	sentence_lengths.append(len(s2.split()))

	for word in s1.split():
		if not (word in word_to_id):
			word_to_id[word] = curr_id
			if word in glove_dict:
				init_glove_matrix[curr_id] = glove_dict[word]
			curr_id += 1

	for word in s2.split():
		if not (word in word_to_id):
			word_to_id[word] = curr_id
			if word in glove_dict:
				init_glove_matrix[curr_id] = glove_dict[word]
			curr_id += 1

print "Max sentence length in data ", max_sentence_len
sentence_lengths = np.array(sentence_lengths)
print "Num more than 50 ", np.sum(sentence_lengths>=50)
print "Num more than 60 ", np.sum(sentence_lengths>=60)
if max_sentence_len > 60:
	max_sentence_len = 60 # Can change the choice of this. This is a free parameter too.

# Train, Test lists creation. Test here is technically more like Validation
X_train_1 = []
X_train_2 = []
y_train = []
X_test_1 = []
X_test_2 = []
y_test = []

train_pc = 0.8
num_train = int(np.ceil(train_pc*len(data_tuples)))
random.seed(186) # Fixing random seed for reproducibility
random.shuffle(data_tuples)

# TRAIN - TEST SPLIT OF THE TUPLES
train_data_tuples = data_tuples[0: num_train]
test_data_tuples = data_tuples[num_train:]
print "Num of training examples ", len(train_data_tuples)

X_train_1, X_train_2, y_train = create_data_matrices(train_data_tuples)
X_test_1, X_test_2, y_test = create_data_matrices(test_data_tuples)
print "Created Training and Test Matrices, and corresponding label vectors"

# create the model
embedding_vecor_length = 300
num_vocab = total_num_words + 1 # since the <none> token is extra
model = Sequential()
model.add(Embedding(input_dim=num_vocab, output_dim=embedding_vecor_length, weights=[init_glove_matrix]))
model.add(LSTM(100, dropout_W=0.5, dropout_U=0.5))
print "Done building core model"

# Inputs to Full Model
input_dim = max_sentence_len
input_1 = Input(shape=(input_dim,))
input_2 = Input(shape=(input_dim,))

# Send them through same model (weights will be thus shared)
processed_1 = model(input_1)
processed_2 = model(input_2)

print "Going to merge the two branches at model level"

merged = merge([processed_1, processed_2], mode='concat')

# Add an FC layer before the Clf layer (non-lin layer after the lstm 'thought vecs' concatenation)
merged_fc = Dense(100, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001), name='merged_fc')(merged)
merged_fc_drop = Dropout(0.4)(merged_fc) # Prevent overfitting at the fc layer

main_output = Dense(1, activation='sigmoid', name='main_output')(merged_fc_drop)

full_model = Model( input=[input_1, input_2], output=main_output )

full_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(full_model.summary())

#saves the model weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint(filepath="models/weights-{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_acc', verbose=1, save_best_only=False)

full_model.fit( [X_train_1, X_train_2], y_train, validation_data=([X_test_1, X_test_2], y_test), nb_epoch=12, batch_size=128, verbose=1, callbacks=[checkpointer])

# Final evaluation of the model
scores = full_model.evaluate( [X_test_1, X_test_2], y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Currently this is only used to count the number of question pairs with at least one empty sentence
# May think of more useful analysis on the processed tuples at a later point.
import numpy as np
import cPickle

with open("data/data_tuples_glovem.p", "rb") as f:
	data_tuples = cPickle.load(f)

cnt = 0
for idx, tx in enumerate(data_tuples):
	s1 = tx[0]
	s2 = tx[1]
	if len(s1.split()) ==0 :
		#print idx, "s1", s1
		cnt += 1
	if len(s2.split()) == 0:
		#print idx, "s2", s2
		cnt += 1

print "Empty sentences num ", cnt

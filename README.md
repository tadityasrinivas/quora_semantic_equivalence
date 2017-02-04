# Semantically Equivalent Question Pair Detection (Quora Question Pairs Dataset)

## Dataset  
These experiments were specifically written to solve semantic duplicate pair detection on the [Question Pairs dataset](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)  

## Packages  
These experiments were written using Keras (with Tensorflow as backend). All the models were trained in CPU mode.  
- Keras  
- Tensorflow  
- h5py (to save models and weights to hdf5 files)  

## Cleaning  
As one would expect with datasets from the wild, there is some amount of cleaning necessary to preprocess the dataset, before any useful learning can be attempted with success. Some things used are: [Spell correction](http://norvig.com/spell-correct.html), Known vocabulary of 1.9 M Glove words (from Common Crawl 42B tokens, found [here](http://nlp.stanford.edu/projects/glove/)), Rules found by analyzing samples from data (such as separating "25kg" into "25 kg"). These steps are implemented in analyze\_data.py.  

Some interesting remaining words, after cleaning (mostly badly spelt words, unintelligible words, long numbers and proper nouns):  ['netskbuffdatausesoffset', 'dhawans', 'magizhchi', 'venkatamangalam', '442301']. It will help to use a more powerful spell checker (also, using a larger set of known words in the spell checker, so that slightly misspelt proper nouns can be spell-corrected).  

## Challenging Nature of Data  

Some examples of noisy labels in the dataset: (format: (s1, s2, label))  
1) ('Why are the powers of the Indian President are unknown/less,  compared to prime minister?', 'In practice, is the president or the prime minister the supreme power in Indian democracy?', 1)  
2) ('Is mint poisonous to cats?', 'Why are mints poisonous to cats?', 1)  
3) ('What are short success motivational stories?', 'What are some short motivation stories?', 0)  
4) ('Does Lipton green tea Assist in weight loss?', 'At what time should I drink green tea to be fit?', 1)  
5) ('How do I control masturbation?', 'What are the best ways to stop or reduce the frequency of masturbation?', 0)  

Somebody else might feel that some of the above are actually correct labels, which indicates that it is sometimes hard for even humans to tell apart semantically duplicate pairs from non-duplicate pairs.  

Some impossible to solve question pairs (due to illegible characters):  
1) ('What does "å°å»ºæˆåŠŸ" mean in Chinese?',	'What does \U00024d13 mean in Chinese?', 0)  
2) ('What is the etymology of é˜¿?', 'What is the etymology of entrÃ©e?', 0)  

Some incredibly hard true positive duplicate pairs (since they need world knowledge)  
1) ('How will issuing of new 2000 Rs notes help curb black money and corruption?', 'How will banning the present 500/1000 notes fight black money if it can be exchanged?', 1)  

This can only be solved if the context of the demonetization exercise in India is known from an external knowledge source, and that issuing new Rs. 2000 notes is equivalent to banning 500 and 1000 notes. It is impossible to solve these kind of examples without aid from external knowledge.   

## Approaches  
Broadly,  
1) Off-the-shelf word vectors + Bag-of-Words/Bag-of-Characters (for unknown words not in Glove)  
2) Learning end-to-end LSTM based model for classification  

## Preliminary Results  
The data was shuffled and split into 80%-20% (train-val) subsets. The following results were obtained with barely any hyperparameter tuning (given the limitations of my current hardware)  
1) Difference of averaged Glove + BoC, Linear SVM = 65.27% Val Accuracy, 65.09% Training Accuracy  
2) Concatenated averaged Glove + BoC, Linear SVM = 70.7% Val Accuracy, 70.6% Training Accuracy   
3) Two LSTM branches + Merge + FC + ReLU + FC (classifier) + Sigmoid =  **78.38%** Val Accuracy, 82.31% Training Accuracy   

In approach 1, a sentence pair was represented as the difference of averaged Glove vectors of its words. In this approach, unknown words were represented using a Bag-of-Characters model, instead of using the unknown token in the Glove dictionary or ignoring them altogether. It is evident that Linear SVM on top of a combination of simple/off-the-shelf features is clearly underfitting the task at hand. This follows from the fact that the training and validation accuracies are about the same. It is not surprising since the model capacity is fairly low, for e.g. there is no way for the non-linear relations to be captured.  

In approach 2, instead of using a difference, the features for the two sentences were simply concatenated. This resulted in an observable jump in the classification accuracy.  

Using two LSTM (approach 3) branches and concatenating the outputs, followed by a classifier significantly improves the results. Embeddings were initialized with Glove vectors (but this initialization didn't seem to matter much). Currently, the hyperparameters used seem to overfit unless early stopping is used (more training results in further divergence of gap between train & val performances). I have experimented with using 2-layer stacked LSTMs and 2 FC layers (2 non-linearities before classifier), but didn't see much improvement (it may be a case of not enough hyperparameter search, since this model was considerably slower to train)  

Besides accuracy, it would also make sense to gauge model performance using Precision/Recall/F-score for the task of duplicate detection. I haven't done this yet, primarily because I'm fairly certain that the cost function will have to be changed for optimal F-score (currently, both positives and negatives are weighted equally, since we use a Sigmoid Cross-Entropy loss. However, we have to readjust the weights for the Positive and Negative terms in this loss, depending on the ratio of Positives and Negatives in the training data.)  

import csv 
import numpy as np
import sklearn as sk
import matplotlib
from matplotlib import pyplot as plt
import nltk 
import keras
import string
import pandas as pd



def listize(filename):
	ratings = []
	dates= []
	variations = []
	verified_reviews = []
	feedbacks = []
	with open(filename, 'r') as file:
		reader = csv.reader(file, delimiter = '\t')
		next(file)
		for rating, date, variation, verified_review, feedback in reader:
			ratings.append(rating)
			dates.append(date)
			variations.append(variation)
			verified_reviews.append(verified_review)
			feedbacks.append(feedback)

	#print(ratings)
	ratings = list(map(int, ratings))
	feedbacks = list(map(int, feedbacks))
	#print(ratings)
	ratings = keras.utils.to_categorical(ratings)
	ratings = np.delete(ratings, 0, 1)
	return ratings, dates, variations, verified_reviews, feedbacks

ratinglist, datelist, variationlist, reviewlist, feedbacklist = listize("amazon_alexa.tsv")
"""
sentencedreviews = []
for review in verified_reviews:
	review = unicode(review, 'utf-8')
	sentencedreviews.append(nltk.sent_tokenize(review))
print(sentencedreviews[2])
"""
#print(verified_reviews)
words = []
for review in reviewlist:
	review = unicode(review, 'utf-8')
	words.append(nltk.word_tokenize(review))
#print words
lengths = []
for review in words: 
	count = 0
	for word in review:
		count += 1
	lengths.append(count)
#print words[0]
#print(lengths[0])
#plt.scatter(lengths, ratings)
#plt.show()
#print(words[2])
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english')) 
#print(lemmatizer.lemmatize('loved', 'v'))
#print(lemmatizer.lemmatize('as'))
#https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''

#print(nltk.pos_tag(words[1]))
#print(lemmatizer.lemmatize('loving', 'v'))

for review in words:
	parts_of_speech = nltk.pos_tag(review)
	for i in range(len(review)): 
		pos = get_wordnet_pos(parts_of_speech[i][1])
		if(pos != ''):
			review[i] = lemmatizer.lemmatize(review[i], pos)
	review[:] = [word for word in review if word not in stop_words]
""" temporary fix - just fix tokenization instead of the following:"""
for review in words: 
	index = len(review)-1
	while (index > -1):
		review[index]=review[index].lower()
		for char in review[index]:
			if(char in string.punctuation):
				review.pop(index)
				break
		index -= 1

#print(words[2])
#words[2][:] = [word for word in words[2] if word not in stop_words]
#print(words[2])

""" BAG OF WORDS """


def create_bag(words):
	bag_of_words = []
	for review in words:
		for word in review:
			if word not in bag_of_words:
				bag_of_words.append(word)
	#print(bag_of_words)
	print(len(bag_of_words))
	with open('bagofwords.txt', 'w') as file:  
	    for word in bag_of_words:
	        file.write('%s\n' % word.encode('UTF-8'))
	return bag_of_words

bag_of_words = create_bag(words)


def vectorize(words):
	arr = []
	for r in range(len(words)):
		arr.append([])
		for word in words[r]:
			arr[r].append(bag_of_words.index(word))
	arr = np.asarray(arr)
	return keras.preprocessing.sequence.pad_sequences(arr, 200	)

max_features = len(bag_of_words)
	#print(X)
Y = ratinglist
X = vectorize(words)
#print(Y)
#print(len(X)==len(Y))
trainvocab, testvocab, trainlabels, testlabels = sk.model_selection.train_test_split(X, Y, shuffle=True, test_size = 0.2)
batch_size = 16
maxlen = 200
trainvocab = keras.preprocessing.sequence.pad_sequences(trainvocab, maxlen=maxlen)
testvocab = keras.preprocessing.sequence.pad_sequences(testvocab, maxlen=maxlen)

trainvocab = np.asarray(trainvocab)
testvocab = np.asarray(testvocab)
trainlabels = np.asarray(trainlabels)
testlabels = np.asarray(testlabels)

print('x_train shape:', trainvocab.shape)
print('x_test shape:', testvocab.shape)
print('y_train shape:', trainlabels.shape)
print('y_test shape:', testlabels.shape)


lossw = []
for i in range(5): 
	count = 0
	for label in trainlabels:
		if(label[i] == 1):
			count += 1
	lossw.append(float(len(trainlabels)/float(count)))


model = keras.Sequential()
model.add(keras.layers.Embedding(max_features, 128))
model.add(keras.layers.LSTM(128, dropout=0.5, recurrent_dropout=0.5))
model.add(keras.layers.Dense(5, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
"""
fives = 0
for arr in trainlabels: 
	if(arr[4]==1):
		fives += 1

avgtrain = fives/float(len(trainlabels))

fives = 0
for arr in testlabels: 
	if(arr[4]==1):
		fives += 1

avgtest = fives/float(len(testlabels))
"""
seed = 10
np.random.seed(seed)
"""
kfold = sk.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
bestscore = 0
count = 0
for train, test in kfold.split(X, Y):
  # create model
  	if(count == 0):
  		Y = keras.utils.to_categorical(Y)
  		Y = np.delete(Y, 0, 1)
  	for i in range(len(X)):
  		X[i] = np.asarray(X[i])
	model = keras.Sequential()
	X = keras.preprocessing.sequence.pad_sequences(X, maxlen=maxlen)
	model.add(keras.layers.Embedding(max_features, 128))
	model.add(keras.layers.LSTM(128, dropout=0.5, recurrent_dropout=0.5))
	model.add(keras.layers.Dense(5, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
	# Fit the model
	model.fit(X[train], Y[train], batch_size=batch_size, epochs=5, verbose=0)
				# evaluate the model
 	scores = model.evaluate(X[test], Y[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
	if (scores[1]*100 > bestscore):
		model.save_weights("crosseval5epochs.h5")
		print("Saved model to disk")
		bestscore = scores[1]*100
	count += 1
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


"""
"""
print('Train...')
model.fit(trainvocab, trainlabels,
          batch_size=batch_size,
          epochs=15,
          validation_data=(testvocab, testlabels), class_weight = lossw)
score, acc = model.evaluate(testvocab, testlabels,
                            batch_size=batch_size)

#print('Test score:', score)
print('Test accuracy:', acc)

# serialize weights to HDF5
model.save_weights("15epochslossweights.h5")
print("Saved model to disk")
"""
"""
10 epochs:
acc: 91.45%
Saved model to disk
acc: 90.85%
acc: 91.52%
Saved model to disk
acc: 90.13%
acc: 91.33%
acc: 91.68%
Saved model to disk
acc: 91.37%
acc: 91.57%
acc: 91.82%
Saved model to disk
acc: 93.61%
Saved model to disk
91.53% (+/- 0.83%)
"""

words = [["love", "it"], ["the", "alexa", "is", "bad"], ["it", "does", "not", "work"], ['bad'], ['work'], ['not', 'work'], ['not', 'bad'], ['alexa']]
#words = [['the', 'voice', 'keeps', 'me', 'happy', 'even', 'though', 'i', 'alone']]
arr = vectorize(words)
model.load_weights("15epochslossweights.h5")
print(model.predict(arr))


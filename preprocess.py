import sys
import pandas as pd
import numpy as np
import re
import nltk

#Data information
sentenceMax = 30
wordVecSize = 50
files = [sys.argv[1], sys.argv[2]]

#Load files and remove any duplicate headlines
clickbait_df = pd.read_csv(files[0])
news_df = pd.read_csv(files[1])
news_df = news_df.drop_duplicates(['headline'])
clickbait_df = clickbait_df.drop_duplicates(['headline'])
print("The unique values of categories are ")
print(clickbait_df['article_category'].unique())

#Below removes foreign languages
ignored_cats = ['Espanol','France','Germany','Brasil','USNews','World']
clickbait_df = clickbait_df.loc[~clickbait_df['article_category'].isin(ignored_cats)]

#load glove vectors
print("Loading glove vectors")
vectors_df = pd.read_csv('GloVeData/glove.6B.50d.txt',sep=' ',header=None,quoting=3)
vectors_df = vectors_df.set_index([0])
print("Vectors loaded")

def textCleaning(df, headlineName,ans):
	#df[headlineName] = df[headlineName].str.replace(r'[0-9]+','#NUM')
	df[headlineName] = df[headlineName].str.lower().str.strip()
	sentenceLength = 0	
	for i, row in df.iterrows():
		tokens = nltk.word_tokenize(df.loc[i, headlineName])
		df.set_value(i,headlineName,tokens)
		if sentenceLength < len(tokens):
			sentenceLength = len(tokens)
	print(sentenceLength)
	df['ans']=np.ones(df.shape[0])*ans
	return(df)
	
def gloveify(df, headlineName, vectors):
		corpora = np.zeros((df.shape[0],sentenceMax*wordVecSize));
		count = 0
		for i, row in df.iterrows():
			sentence = np.zeros((sentenceMax,wordVecSize))
			for j, word in enumerate(df.loc[i, headlineName]):
				if word in vectors.index:
					sentence[j,:]=vectors.loc[word, :].values
				else:
					sentence[j,:]=np.ones(wordVecSize)	
			corpora[count,:]=sentence.flatten()
			count = count+1
		return(corpora)
		
print("Cleaning text")
clickbait_df = textCleaning(clickbait_df, 'headline',1)
news_df = textCleaning(news_df, 'headline',0)	
print("Text cleaned")

vocabulary=set()
clickbait_df['headline'].apply(vocabulary.update)
news_df['headline'].apply(vocabulary.update)

print("Vectorising text")
dataMat = gloveify(clickbait_df, 'headline',vectors_df)
dataMat = np.concatenate((dataMat,gloveify(news_df, 'headline',vectors_df)),axis=0)
print("Text vectorised")
print(dataMat)

labels = pd.concat([clickbait_df.loc[:,'ans'],news_df.loc[:,'ans']])

print('Shuffling and splitting')
assert len(labels.as_matrix()) == len(dataMat)
rand_index = np.random.permutation(len(dataMat))
dataMat = dataMat[rand_index]
labelMat = labels.as_matrix()[rand_index]
print('Done shuffling, now saving')

split = int(len(dataMat)*0.97)

np.savetxt('dataNEWX.txt',dataMat[0:split,:] , fmt='%f')
np.savetxt('labelsNEWX.txt', labelMat[0:split], fmt='%d')

np.savetxt('testData.txt',dataMat[split:len(dataMat),:] , fmt='%f')
np.savetxt('testLabels.txt', labelMat[split:len(dataMat)], fmt='%d')

with open('vocabNEW.txt', 'w') as f:
	for word in list(vocabulary):
		f.write(word+'\n')

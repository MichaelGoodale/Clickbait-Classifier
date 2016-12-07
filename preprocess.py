import sys
import pandas as pd
import numpy as np
import nltk

#File Paths
GLOVE_FILE = "GloVeData/glove.6B.50d.txt"
DATA_PATH = "ProcessedData/"
DATA_FILE = DATA_PATH + "data.txt"
LABEL_FILE = DATA_PATH + "labels.txt"
TEST_DATA_FILE = DATA_PATH + "testData.txt"
TEST_LABEL_FILE = DATA_PATH + "testLabels.txt"

#Data information
SENTENCE_MAX = 30
WORD_VECTOR_SIZE = 50
TRAIN_TEST_RATIO = 0.95
files = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]

#Load files and remove any duplicate headlines
buzzfeed_df = pd.read_csv(files[0])
news_df = pd.read_csv(files[1])
news_df = news_df.drop_duplicates(["headline"])
buzzfeed_df = buzzfeed_df.drop_duplicates(["headline"])

reddit_df = pd.read_csv(files[2], sep="\n")
viral_df = pd.read_csv(files[3], sep="\n")
reddit_df.columns = ["headline"]
viral_df.columns = ["headline"]

#Remove foreign languages, and news-ier titles
ignored_cats = ["Espanol", "France", "Germany", "Brasil", "USNews", "World"]
buzzfeed_df = buzzfeed_df.loc[~buzzfeed_df["article_category"].isin(ignored_cats)]

#load GloVe vectors
print("Loading GloVe vectors")
vectors_df = pd.read_csv(GLOVE_FILE, sep=" ", header=None, quoting=3)
vectors_df = vectors_df.set_index([0])
print("Vectors loaded")

def textCleaning(df, headline_name, ans):
    df[headline_name] = df[headline_name].str.lower().str.strip()
    max_sentence_length = 0
    for i, row in df.iterrows():
        tokens = nltk.word_tokenize(df.loc[i, headline_name])
        df.set_value(i, headline_name, tokens)
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)
    print(max_sentence_length)
    df["ans"] = np.ones(df.shape[0])*ans
    return(df)

def gloveify(df, headline_name, vectors):
    corpus = np.zeros((df.shape[0], SENTENCE_MAX*WORD_VECTOR_SIZE))
    count = 0
    for i, row in df.iterrows():
        sentence = np.zeros((SENTENCE_MAX, WORD_VECTOR_SIZE))
        if len(df.loc[i, headline_name]) > SENTENCE_MAX:
            continue
        for j, word in enumerate(df.loc[i, headline_name]):
            if word in vectors.index:
                sentence[j, :] = vectors.loc[word, :].values
            else:
                sentence[j, :] = np.ones(WORD_VECTOR_SIZE)
        corpus[count, :] = sentence.flatten()
        count = count+1
    return(corpus)

print("Cleaning text")
buzzfeed_df = textCleaning(buzzfeed_df, "headline", 1)
news_df = textCleaning(news_df, "headline", 0)
reddit_df = textCleaning(reddit_df, "headline", 0)
viral_df = textCleaning(viral_df, "headline", 1)
print("Text cleaned")

print("Vectorising text")
buzzMat = gloveify(buzzfeed_df, "headline", vectors_df)
newsMat = gloveify(news_df, "headline", vectors_df)
redditMat = gloveify(reddit_df, "headline", vectors_df)
viralMat = gloveify(viral_df, "headline", vectors_df)
dataMat = np.concatenate((buzzMat, newsMat, redditMat, viralMat), axis=0)
print("Text vectorised")

labels = pd.concat([buzzfeed_df.loc[:, "ans"], news_df.loc[:, "ans"], reddit_df.loc[:, "ans"], viral_df.loc[:, "ans"]])

print("Shuffling and splitting")
assert len(labels.as_matrix()) == len(dataMat)
rand_index = np.random.permutation(len(dataMat))
dataMat = dataMat[rand_index]
labelMat = labels.as_matrix()[rand_index]
print("Done shuffling, now saving")

split = int(len(dataMat)*TRAIN_TEST_RATIO)

np.savetxt(DATA_FILE, dataMat[0:split, :], fmt="%f")
np.savetxt(LABEL_FILE, labelMat[0:split], fmt="%d")

np.savetxt(TEST_DATA_FILE, dataMat[split:len(dataMat), :], fmt="%f")
np.savetxt(TEST_LABEL_FILE, labelMat[split:len(dataMat)], fmt="%d")

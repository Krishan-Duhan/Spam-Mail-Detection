from numpy import *
import os
import re
import sys
from collections import Counter
from math import log
from math import e

# This function returns the list of stopwords in a given file
def extract_stopwords(stopwords_file):
    stop_words = []
    fil = open(stopwords_file)
    stop_words = fil.read().strip().split()
    return stop_words

# Function to read data files by removing stopWords from them
def read_withoutSW(folder, stopWordsF):
    files = os.listdir(folder)

    diction = {}
    vocab = []
    stop_words = extract_stopwords(stopWordsF)
    for f in files:
        fil = open(folder + "/" + f, encoding = "ISO-8859-1")
        words = fil.read()                                      # Read all text in a single string
        words = re.sub('[^a-zA-Z]',' ', words)                  # [^... ] matches any single char not in brackets
        words_all = words.strip().split()                       # strip removes spaces
        req_words = []                                          # Required words will not contain stopwords

        for wd in words_all:
            if(wd not in stop_words):
                req_words.append(wd)
        diction[f] = req_words
        vocab.extend(req_words)
    return vocab, diction

# Function to read data files without removing stopWords
def read_withSW(folder):
    files = os.listdir(folder)
    diction = {}
    vocab = []
    for f in files:
        fil = open(folder + "/" + f,encoding = "ISO-8859-1")
        words = fil.read()                                      # Read all text in a single string
        #words = re.sub('[^a-zA-Z]',' ', words)i                # [^... ] matches any single char not in brackets
        words_all = words.strip().split()                       # strip removes spaces
        diction[f] = words_all
        vocab.extend(words_all)
    return vocab, diction


# Function to train for MCAP Logistic Regression & returns the weight vector
def trainM_LR(train_features, label_list, lambdaa):
    feature_mat = mat(train_features)                           # list of lists converted to matrix
    p,q = shape(feature_mat)
    label_mat = mat(label_list).transpose()
    eeta = 0.1
    wt = zeros((q,1))
    num_iteration = 100
    for i in range(num_iteration):
        predict_condProb = 1.0/(1 + exp(-feature_mat*wt))
        err = label_mat - predict_condProb
        wt = wt + eeta*feature_mat.transpose()*err - eeta*lambdaa*wt
    return wt


# Function to apply MCAP Logistic Regression on given test set and returns the accuracy
def applyM_LR(wt, test_features, len_test_spamDict, len_test_hamDict):
    feature_mat = mat(test_features)
    res = feature_mat*wt
    corr = 0
    len_allDict = len_test_spamDict + len_test_hamDict

    for i in range(len_test_spamDict):
        if(res[i][0] < 0.0):
            corr += 1
    i = 0
    for i in range(len_test_spamDict+1, len_allDict):
        if(res[i][0] > 0.0):
            corr += 1
    return (float)(corr/len_allDict)

def feature_vector(all_distinctWords, diction):
    feature = []
    for f in diction:
        row = [0]*(len(all_distinctWords))
        for term in all_distinctWords:
            if(term in diction[f]):
                row[all_distinctWords.index(term)] = 1
        row.insert(0,1)                                 # Making X0 = 1 in feature vector
        feature.append(row)
    return feature                                      # Returning list of lists

if __name__ == "__main__":
    if(len(sys.argv) != 7):
        print("Incorrect arguments passed !!")
        print("Correct format is: ./LogReg.py <spam_trainPath> <ham_trainPath> <spam_testPath> <ham_testPath>",
                "<stopWords file path> <yes/no to remove stopwords>")
        sys.exit()

    train_spam = sys.argv[1]
    train_ham = sys.argv[2]
    test_spam = sys.argv[3]
    test_ham = sys.argv[4]
    stopWords = sys.argv[5]
    ThrowStopWords = sys.argv[6]
    lambdaa = 0.1

    if(ThrowStopWords == "yes"):
        train_spamVocab, train_spamDict = read_withoutSW(train_spam, stopWords)
        train_hamVocab, train_hamDict = read_withoutSW(train_ham, stopWords)
    else:
        train_spamVocab, train_spamDict = read_withSW(train_spam)
        train_hamVocab, train_hamDict = read_withSW(train_ham)

    test_spamVocab, test_spamDict = read_withSW(test_spam)
    test_hamVocab, test_hamDict = read_withSW(test_ham)

    all_distinctWords = list(set(train_spamVocab)|set(train_hamVocab))
    all_trainDict = train_spamDict.copy()
    all_trainDict.update(train_hamDict)

    all_testDict = test_spamDict.copy()
    all_testDict.update(test_hamDict)

    label_list = []
    for i in range(len(train_spamDict)):
        label_list.append(0)
    i = 0
    for i in range(len(train_hamDict)):
        label_list.append(1)

    train_features = feature_vector(all_distinctWords, all_trainDict)
    test_features = feature_vector(all_distinctWords, all_testDict)
    
    wt = trainM_LR(train_features, label_list, lambdaa)
    accuracy = applyM_LR(wt, test_features, len(test_spamDict), len(test_hamDict))
    print("Accuracy of Logistic Regression is: ", accuracy)

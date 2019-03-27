# Pseudo-code shared in Fig 13.2 here:  http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
import os
import re
import sys
from collections import Counter
from math import log
from math import e

all_distinctWords = []

# Function to train as per Naive Bayes model and return prior_probability, conditional_probability
def trainM_NB(num_spamdocs, num_hamdocs, train_spamVocab, train_hamVocab):
    global all_distinctWords
    prior_spam = (float)(num_spamdocs/(num_spamdocs + num_hamdocs))      # prior probability of spam class
    prior_ham =(float)(num_hamdocs/(num_spamdocs + num_hamdocs))
    
    all_spam_dict = Counter(train_spamVocab)                    #spam dictionary from all spam docs in training data, contains "word:count"
    all_ham_dict = Counter(train_hamVocab)
    
    numWords_spam = len(train_spamVocab)                        # total num of words in all spam docs
    numWords_ham = len(train_hamVocab)
    
    all_distinctWords = list(set(all_spam_dict)|set(all_ham_dict))          #set contains only set of distinct words, counts removed. | = join
    num_distinctWords = len(all_distinctWords)
    
    condprob_spam = {}                                          # dictionary to hold conditional probability for words in spam files
    condprob_ham = {}

    for term in all_distinctWords:
        count = 0
        if term in all_spam_dict:
            count = all_spam_dict[term] 
        cond_probS = (float)((count + 1)/(numWords_spam + num_distinctWords))
        condprob_spam[term] = cond_probS

    for term in all_distinctWords:
        count = 0
        if term in all_ham_dict:
            count = all_ham_dict[term]
        cond_probH = (float)((count + 1)/(numWords_ham + num_distinctWords))
        condprob_ham[term] = cond_probH

    return prior_spam, prior_ham, condprob_spam, condprob_ham

# Function to apply Naive Bayes on given test sets
def applyM_NB(prior_spam, prior_ham, condprob_spam, condprob_ham, test_spamDict, test_hamDict):
    global all_distinctWords
    spam_hamDict = [test_spamDict, test_hamDict]
    corr = 0
    for i in range(len(spam_hamDict)):
        for f in spam_hamDict[i]:
            s1 = log(prior_spam)
            s2 = log(prior_ham)
            for term in spam_hamDict[i][f]:
                if term in all_distinctWords:
                    s1 += log(condprob_spam[term])
                    s2 += log(condprob_ham[term])
            if(s1 >= s2 and i == 0):
                corr += 1
            elif(s1 <= s2 and i == 1):
                corr += 1
    return (float)(corr/(len(test_spamDict) + len(test_hamDict)))

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


if __name__ == "__main__":
    if(len(sys.argv) != 7):
        print("Incorrect arguments passed !!")
        print("Correct format is: ./NaiveBayes.py <spam_trainPath> <ham_trainPath> <spam_testPath> <ham_testPath>", 
                "<stopWords file path> <yes/no to remove stopwords>")
        sys.exit()

    train_spam = sys.argv[1]
    train_ham = sys.argv[2]
    test_spam = sys.argv[3]
    test_ham = sys.argv[4]
    stopWords = sys.argv[5]
    ThrowStopWords = sys.argv[6]

    if(ThrowStopWords == "yes"):
        train_spamVocab, train_spamDict = read_withoutSW(train_spam, stopWords)
        train_hamVocab, train_hamDict = read_withoutSW(train_ham, stopWords)
    else:
        train_spamVocab, train_spamDict = read_withSW(train_spam)
        train_hamVocab, train_hamDict = read_withSW(train_ham)

    test_spamVocab, test_spamDict = read_withSW(test_spam)
    test_hamVocab, test_hamDict = read_withSW(test_ham)

    num_spamDocs = len(train_spamDict)              # total no of spam docs in training data
    num_hamDocs = len(train_hamDict)

    prior_spam, prior_ham, condprob_spam, condprob_ham = trainM_NB(num_spamDocs, num_hamDocs, train_spamVocab, train_hamVocab)

    accuracy = applyM_NB(prior_spam, prior_ham, condprob_spam, condprob_ham, test_spamDict, test_hamDict)
    print("Accuracy of Naive bayes = ", accuracy)

from nltk.stem.porter import PorterStemmer
import numpy as np
import nltk

Stemmer = PorterStemmer()

def Tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return Stemmer.stem(str(word).lower())

def bagOfWords(TokenizedSentence, words):
    sentence_word = [stem(word) for word in TokenizedSentence]
    Bag = np.zeros(len(words), dtype = np.float64)
    for idx , w in enumerate(words):
        if w in sentence_word:
            Bag[idx] = 1

        return Bag
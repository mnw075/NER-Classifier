# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 02:22:41 2019

@author: mwon579
"""

# import libraries
import os
import pandas as pd
import nltk, re, pprint
import random
from nltk.classify import apply_features
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.corpus import conll2000 
from nltk.tag import pos_tag  
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

'''
Experimenting with using dictionaries instead of pandas
'''
#def gender_features(word):
#    return {'last_letter': word[-1]}
# 	
#from nltk.corpus import names
#labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
#[(name, 'female') for name in names.words('female.txt')])
#random.shuffle(labeled_names)

#featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
#train_set, test_set = featuresets[500:], featuresets[:500]
#gender_classifier = nltk.NaiveBayesClassifier.train(train_set)
#print(nltk.classify.accuracy(gender_classifier, test_set))
#
#nonceosp = pd.read_csv('nonceo.csv',sep = '\t',encoding='ANSI', header=None)
#nonceos = []
#for i in range(len(nonceosp)):
#    nonceos.append(nonceosp.values[i][0])
#    
#ceosp = pd.read_csv('ceo.csv',sep = '\t',encoding='ANSI', header=None)
#ceos = []
#for i in range(len(ceosp)):
#    ceos.append(ceosp.values[i][0])
#
#labeled_ceos = ([(name, 'ceo') for name in ceos] +
#[(name, 'nonceo') for name in nonceos])
#random.shuffle(labeled_ceos)
#vowels = list("aeiouy")
#consonants = list("bcdfghjklmnpqrstvexz")
#	
#def ceo_features(name):
#    features = {}
#    features["first_letter"] = name[0].lower()
#    features["last_letter"] = name[-1].lower()
#    features["length"] = len(name)
#    number_of_consonants = sum(name.count(c) for c in consonants)
#    number_of_vowels = sum(name.count(c) for c in vowels)
#    features['number_of_consonants'] = number_of_consonants
#    features['number_of_vowels'] = number_of_vowels
#    features['more_vowels'] = number_of_consonants <= number_of_vowels
#
#    for letter in 'abcdefghijklmnopqrstuvwxyz':
#        features["count({})".format(letter)] = name.lower().count(letter)
#        features["has({})".format(letter)] = (letter in name.lower())
#
#featuresets = [(ceo_features(n), label) for (n, label) in labeled_ceos]
#train_set, test_set = featuresets[2200:], featuresets[:2200]
#
#
#train_set = apply_features(ceo_features, labeled_ceos[2200:])
#test_set = apply_features(ceo_features, labeled_ceos[:2200])
#classifier = nltk.NaiveBayesClassifier.train(train_set)
#print(nltk.classify.accuracy(classifier, test_set))


'''
Reading in Data
'''
ceosp = pd.read_csv('ceo.csv',sep = '\t',encoding='ANSI', header=None)
ceos = []
for i in range(len(ceosp)):
    ceos.append(ceosp.values[i][0])

ceos_parsed = []
for ceo in ceos:
    tokens = re.split(r',',ceo)
    if (tokens[0]) != '' and (tokens[0][-1] == ' '):
            tokens[0] = tokens[0][:-1]
    name = ''
    for i in range(len(tokens)):   
        if (i == 0) and (tokens[0] != ''):
            name += tokens[i] + ' '
        else:
            name += tokens[i]
    ceos_parsed.append(name)
ceos = set(ceos_parsed)

companiesp = pd.read_csv('companies.csv',sep = '\t',encoding='ANSI', header=None)
companies = []
for i in range(len(companiesp)):
    companies.append(companiesp.values[i][0])
companies = set(companies)

corpus2013 = []
corpus2014 = []

for file in os.listdir('2013'):
    corpus2013.append(open(os.path.join('2013',file),'rb').read().decode('ansi'))

for file in os.listdir('2014'):
    corpus2014.append(open(os.path.join('2014',file),'rb').read().decode('ansi'))


# tokenize text - remember to convert text to lower case
#percent_tokens2013 = [percent_tokenizer.tokenize(article.lower()) for article in corpus2013]
#
#ceo_tokenizer = RegexpTokenizer("[A-Z][A-Za-z]+ ?, ?[A-Z][A-Za-z]+|[A-Z][A-Za-z]+ [A-Z][A-Za-z]+?\.? ?[A-Z][A-Za-z]+")
#
#ceo_tokens2013 = [percent_tokenizer.tokenize(article) for article in corpus2013]

'''
Sentence tokenizing, word tokenizing, pos tagging words
'''
pos_taggedSentences2013 = []    
for document in corpus2013:    
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    for sent in sentences:
        pos_taggedSentences2013.append(sent)


'''
Experimenting with chunking and swapping chunk tags
'''
print(nltk.ne_chunk(sentences[0]))
    
grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""
cp = nltk.RegexpParser(grammar)

a = cp.parse(sentences[0])





'''
NP Chunk tag classifier
'''    
class ConsecutiveNPChunkTagger(nltk.TaggerI): 
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) 
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(
            train_set)
    
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): 
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)
    
 	
def npchunk_features(sentence, i, history):
     word, pos = sentence[i]
     if i == 0:
         prevword, prevpos = "<START>", "<START>"
     else:
         prevword, prevpos = sentence[i-1]
     return {"pos": pos, "word": word, "prevpos": prevpos, "prevword": prevword}

'''
NER Chunk tag classifier
'''
import string
from nltk.stem.snowball import SnowballStemmer
 
class ConsecutiveNERChunkTagger(nltk.TaggerI): 
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = features(untagged_sent, i, history) 
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(
            train_set)
    
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNERChunker(nltk.ChunkParserI): 
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

# feature: checck if actual numbers exist in percent token 
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)
 
    # shift the index with 2, to accommodate the padding
    index += 2
 
    word, pos = tokens[index]
    
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    isnumber = is_number(word)
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,
        'number': isnumber,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
 
        'prev-iob': previob,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }

'''
NP chunk first
'''	
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
NP_chunker = ConsecutiveNPChunker(train_sents)    
print(NP_chunker.evaluate(test_sents))  

NP_tagged2013 = []
for sent in pos_taggedSentences2013:
    NP_tagged2013.append(NP_chunker.parse(sent))

#'''
#Change NP to CEO
#'''
#
#stop_words=sorted(set(stopwords.words("english")))
#print(stop_words)
#
#def extract_str(subtree):
#    return ' '.join(word for word, tag in subtree.leaves())
#   
#  
#
#CEO_tagged2013 = NP_tagged2013
#for tagged_sent in CEO_tagged2013:
#    for index, subtree in enumerate(tagged_sent):
#        if (type(subtree) != tuple) and (subtree.label() == 'NP'):
#            s = extract_str(subtree)
#            if any(w for w in ceos if w in s):
#                tagged_sent[index] = nltk.tree.Tree('CEO',subtree[:])
# 
#'''
#Get rid of NP tags and only keep CEO
#
#1) Convert chunked data in IOB format
#2) Remove all [NP] tags and keep only [CEO] tags
#'''
#
#CEO_taggedIOB2013 = []
#for sent in CEO_tagged2013:
#    text = ""
#    sent = nltk.chunk.tree2conlltags(sent)
#    for tup in sent:   
#        for i in range(len(tup)):
#            if i == (len(tup)-1):
#                text += tup[i] + '\n'
#            else:
#                text += tup[i] + ' '
#    CEO_taggedIOB2013.append(text)
#    
#CEO_only_tagged2013 = []
#for iob_sent in CEO_taggedIOB2013:
#    CEO_only_tagged2013.append(nltk.chunk.conllstr2tree(iob_sent, chunk_types=['CEO']))
#
#train_sents = CEO_only_tagged2013[:len(CEO_only_tagged2013)//2]
#test_sents = CEO_only_tagged2013[len(CEO_only_tagged2013)//2:]
#
#CEO_chunker = ConsecutiveNERChunker(train_sents)     
#print(CEO_chunker.evaluate(test_sents))
#
#predicted_CEOs = []
#for sent in pos_taggedSentences2013:
#    CEO_sent = CEO_chunker.parse(sent)
#    for index, subtree in enumerate(CEO_sent):
#        if (type(subtree) != tuple) and (subtree.label() == 'CEO'):
#            predicted_CEOs.append(extract_str(subtree))
#    
#a ='Billy Bob owns a large company.'
#a = nltk.word_tokenize(a)
#a = nltk.pos_tag(a)
#
#print(CEO_chunker.parse(a))
#
#a ='Billy Bob likes to cook food'
#a = nltk.word_tokenize(a)
#a = nltk.pos_tag(a)
#
#print(CEO_chunker.parse(a))

#with open('C:\\Users\\Micha\\AppData\\Roaming\\nltk_data\\corpora\\conll2000\\train2013.txt', 'w') as f:
#    for item in CEO_taggedIOB2013:
#        f.write(u"%s" % item)
  
'''
Full NER
'''
# helper function to untag chunks and extract the string only
def extract_str(subtree):
    return ' '.join(word for word, tag in subtree.leaves())
 
# define regex tokenizer
percent_tokenizer = RegexpTokenizer("-?\w+(?:\.\w+)? %|-?\w+(?:\.\w+)? percent(?:age points?|ile (?:points?)?)?")

#percent_tokenizer.tokenize(corpus2013[2])  

'''
Go thru all NP tags and see if it should be changed to CEO, COMP, or PCT tag
'''
NERNP_tagged2013 = NP_tagged2013
for tagged_sent in NP_tagged2013:
    for index, subtree in enumerate(tagged_sent):
        if (type(subtree) != tuple) and (subtree.label() == 'NP'):
            s = extract_str(subtree)
            if any(w for w in ceos if w in s):
                tagged_sent[index] = nltk.tree.Tree('CEO',subtree[:])
            elif any(w for w in companies if w in s): 
                tagged_sent[index] = nltk.tree.Tree('COMP',subtree[:])
            elif percent_tokenizer.tokenize(s):
                tagged_sent[index] = nltk.tree.Tree('PCT',subtree[:])

'''
Convert tagged sentences to IOB format
'''
NERNP_taggedIOB2013 = []
for sent in NERNP_tagged2013:
#    text = ""
    sent = nltk.chunk.tree2conllstr(sent)
#    for tup in sent:   
#        for i in range(len(tup)):
#            if i == (len(tup)-1):
#                text += tup[i] + '\n'
#            else:
#                text += tup[i] + ' '
    NERNP_taggedIOB2013.append(sent)

'''
Remove all NP tags and keep only the NER tags [CEO,COMP, PCT]
'''
NER_only_tagged2013 = []
for iob_sent in NERNP_taggedIOB2013:
    NER_only_tagged2013.append(nltk.chunk.conllstr2tree(iob_sent, chunk_types=['CEO', 'COMP', 'PCT']))


'''
Train Classifier
'''
# CEO, COMP, PCT, and NP chunks
random.shuffle(NERNP_tagged2013)
train_sents = NERNP_tagged2013[:len(NERNP_tagged2013)//4]
test_sents = NERNP_tagged2013[len(NERNP_tagged2013)//4:]
NERNP_chunker = ConsecutiveNERChunker(train_sents)     
print(NERNP_chunker.evaluate(test_sents))

full_model = ConsecutiveNERChunker(NERNP_tagged2013)

a ='CEO Bill Gates owns Microsoft and gained 20%.'
a = nltk.word_tokenize(a)
a = nltk.pos_tag(a)

print(NERNP_chunker.parse(a))

a ='Billy Bob likes to cook food'
a = nltk.word_tokenize(a)
a = nltk.pos_tag(a)

print(NERNP_chunker.parse(a))

predicted_CEOs = []
predicted_COMPs = []
predicted_PCTs = []
for sent in pos_taggedSentences2013:
    CEO_sent = NERNP_chunker.parse(sent)
    for index, subtree in enumerate(CEO_sent):
        if (type(subtree) != tuple):
            s = extract_str(subtree)
            if (subtree.label() == 'CEO'):
                predicted_CEOs.append(s)
            elif (subtree.label() == 'COMP'): 
                predicted_COMPs.append(s)
            elif (subtree.label() == 'PCT'):
                predicted_PCTs.append(s)

import csv
with open('predicted_ceo.csv','wb') as file:
    for line in predicted_CEOs:
        file.write(line.encode('UTF-8'))
        file.write('\n'.encode('UTF-8'))

with open('predicted_comp.csv','wb') as file:
    for line in predicted_COMPs:
        file.write(line.encode('UTF-8'))
        file.write('\n'.encode('UTF-8'))

with open('predicted_percentage.csv','wb') as file:
    for line in predicted_PCTs:
        file.write(line.encode('UTF-8'))
        file.write('\n'.encode('UTF-8'))

# only CEO, COMP, and PCT chunks
random.shuffle(NER_only_tagged2013)
train_sents = NER_only_tagged2013[:len(NER_only_tagged2013)//4]
test_sents = NER_only_tagged2013[len(NER_only_tagged2013)//4:]

NER_only_chunker = ConsecutiveNERChunker(train_sents)     
print(NER_only_chunker.evaluate(test_sents))

a ='Bill Gates owns Microsoft and gained 20% returns.'
a = nltk.word_tokenize(a)
a = nltk.pos_tag(a)
print(NER_only_chunker.parse(a))

a ='The dog jumped over the brown fence.'
a = nltk.word_tokenize(a)
a = nltk.pos_tag(a)
print(NER_only_chunker.parse(a))


'''
Biased Sampling
'''
positive_samples = []
negative_samples = []
for tagged_sent in NER_only_tagged2013:
    broken = 0
    for index, subtree in enumerate(tagged_sent):
        if (type(subtree) != tuple) and (subtree.label() in ['CEO','COMP','PCT']):
            positive_samples.append(tagged_sent)
            broken = 1
            break
    if broken:
        continue
    negative_samples.append(tagged_sent)

biased_sample = positive_samples + negative_samples[:len(positive_samples)]
    
random.shuffle(biased_sample)
train_sents = biased_sample[:len(biased_sample)//4]
test_sents = biased_sample[len(biased_sample)//4:]
NER_only_chunker = ConsecutiveNERChunker(train_sents)     
print(NER_only_chunker.evaluate(test_sents))
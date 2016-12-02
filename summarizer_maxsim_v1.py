# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 21:35:49 2016

@author: Gaurangi Raul
"""
import os
import mimetypes
import re
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
    
def preprocess(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #print(filtered_sentence)
    ps = PorterStemmer()
    stemmed_sentence = [ps.stem(w) for w in filtered_sentence]
    #print(stemmed_sentence)
    preprocess_sentence = ' '.join([w for w in stemmed_sentence])
    return preprocess_sentence
    
#extract sentences similar to max sentences based on TF-IDF scores
def summarize_maxsim(filename):
    input_file = open(filename, "r", encoding="latin1")
    file_text = input_file.read()
    input_file.close()
    docs = re.split('\. |\? |! |\n',file_text)
    for sent in docs:
        if sent == '':
            docs.remove(sent)
    
    documents = []
    for sent in docs:        
        documents.append(preprocess(sent))
    print("Title:", documents[0])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    #print(tfidf_matrix.shape)
    #print(tfidf_matrix.shape[0])
    sim_scores = cosine_similarity(tfidf_matrix[0:tfidf_matrix.shape[0]], tfidf_matrix)
    scores_dict = {}
    for i in range(1, tfidf_matrix.shape[0]):
        scores_sum = 0
        for j in range(0, tfidf_matrix.shape[0]):
            if sim_scores[i][j] > 0:
                scores_sum += sim_scores[i][j]
        scores_dict[i]=scores_sum
    sorted_scores = sorted(scores_dict.items(), key=operator.itemgetter(1), reverse=True)
    #print(sorted_scores)
    #limit = min(int(0.05*len(scores_dict)), 5)
    limit = int(0.1*len(scores_dict))
    print("Limit:", limit)
    summary = ""
    for i in range(0, limit):
        summary = summary + (docs[sorted_scores[i][0]] + ". ")
    #create output file to write the summary
    out_filename = "C:/USC stuff/Applied NLP/Group Project/Test1/Sys1/abstract" + os.path.basename(filename)
    print(out_filename)
    output_file = open(out_filename, "w", encoding="latin1")
    output_file.write(summary)
    output_file.close()
    

my_directory = "C:/USC stuff/Applied NLP/Group Project/Test1/Docs"
#search through the main directory for text files & execute 1st iteration
for dirpath, dirnames, filenames in os.walk(my_directory):
    for filename in filenames:
        if mimetypes.guess_type(filename)[0] == 'text/plain':
            file = os.path.join(dirpath, filename)
            summarize_maxsim(file)


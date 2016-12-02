# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 23:05:52 2016

@author: Gaurangi Raul
"""
import os
import mimetypes
import re
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
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
    
def get_max_similar_sentence(docs):
    documents = []
    for sent in docs:        
        documents.append(preprocess(sent))
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    sim_scores = cosine_similarity(tfidf_matrix[0:tfidf_matrix.shape[0]], tfidf_matrix)
    scores_dict = {}
    for i in range(0, tfidf_matrix.shape[0]):
        scores_sum = 0
        for j in range(0, tfidf_matrix.shape[0]):
            if sim_scores[i][j] > 0:
                scores_sum += sim_scores[i][j]
        scores_dict[i]=scores_sum
    sorted_scores = sorted(scores_dict.items(), key=operator.itemgetter(1), reverse=True)
    #print(docs)
    #print(documents[sorted_scores[0][0]])
    return docs[sorted_scores[0][0]]


def summarize_clustering(filename):
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
    
    #kmeans Clustering
    #k = min(int(len(sim_scores) * 0.05), 5)
    k = int(len(sim_scores) * 0.1)
    print("Clusters:", k)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(sim_scores)
    #kmeans_centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    #initialize cluster dictionary
    cluster_dict = {}
    for j in range(0, k):
        cluster_dict[j] = []
    order_list = []
    for i in range(0, len(labels)):
        if labels[i] not in order_list:
            order_list.append(labels[i])
            cluster_dict[labels[i]].append(i)
            
    summary = ""
    for i in range(0, k):
        sentence_list = []
        for j in range(0, len(cluster_dict[order_list[i]])):
            sentence_list.append(docs[cluster_dict[order_list[i]][j]])
            summary = summary + get_max_similar_sentence(sentence_list) + ". "
    
    #create output file to write the summary
    out_filename = "C:/USC stuff/Applied NLP/Group Project/Test1/Sys3/abstract" + os.path.basename(filename)
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
            print(file)
            summarize_clustering(file)

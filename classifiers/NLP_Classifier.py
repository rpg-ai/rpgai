# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:35:57 2021
@author: Thiago Russo
"""
# Base Packages
import pickle
import os
# Vectorizer for dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
# NLP Pre processing
from NLP_Text_Preprocessor import NLP_Text_Preprocessor


class NLP_Classifier:
    def __init__(self):
        self.NLP_pp = NLP_Text_Preprocessor()
        pass

    # Used to create a NLP model from scratch
    # Corpus is a list of texts to be analyzed
    def create_TFIDF_Vec_model(self, corpus, path_models):
        # Call NLP pre processing
        clean_text = self.NLP_pp.preprocess(corpus)
        # BOW and TFIDF, term must appear at least a number of times, limit max features and use ngrams
        tfidf_vec = TfidfVectorizer(analyzer = 'word', max_df = 0.90, min_df = 3, ngram_range=(1, 2))
        # Build dictionary to be able to use after training
        tfidf_vec = tfidf_vec.fit(clean_text)
        # Get counts on terms
        bow_tfidf = tfidf_vec.transform(clean_text)
        # Save dictionary
        path = os.path.join(path_models, "tfidf.pickle")
        pickle.dump(tfidf_vec, open(path, "wb"))
        return bow_tfidf

    # Used to score data using NLP from a pre trained model
    def use_TFIDF_Vec_model(self, corpus, path_models):
        # Call NLP pre processing
        clean_text = self.NLP_pp.preprocess(corpus)
        # Load dictionary
        path = os.path.join(path_models, "tfidf.pickle")
        tfidf_vec = pickle.load(open(path,'rb'))
        # Get counts on terms
        bow_tfidf = tfidf_vec.transform(clean_text)
        return bow_tfidf

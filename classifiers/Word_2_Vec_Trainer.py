# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:49:40 2021

@author: Thiago Russo
"""
#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd

import nltk
nltk.download('punkt')

path_data = 'C:\\app\\rpgai\\data\\Dados_Teste.parquet'
# Load data for all models
df_prep = pd.read_parquet(path_data)

df_prep.info()

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(df_prep['backward_text'].to_list())]

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("C:\\app\\rpgai\\classifiers\\models\\doc2vec.model")

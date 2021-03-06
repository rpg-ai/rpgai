# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 18:28:10 2021

@author: Thiago Russo
"""
import flask
from flask import request
import os
import pickle
import time
import numpy as np


import pandas as pd
from NLP_Classifier import NLP_Classifier

app = flask.Flask(__name__)
app.config["DEBUG"] = True
 
with app.app_context():
    global nlp 
    global dp
    global skill_model
    global skill_tfidf
    global path_dir_models
    
    
    nlp = NLP_Classifier()
    dict_models = dict()
    dict_tfidf = dict()
    
    path_dir_models = 'caminho'
    
    filename = os.path.join(path_dir_models, 'model.sav' )
    skill_model = pickle.load(open(filename, 'rb'))

    filename = os.path.join(path_dir_models, 'tfidf.pickle' )
    skill_tfidf = pickle.load(open(filename, 'rb'))
    
    

@app.route('/', methods=['GET'])
def home():
    return "on"


@app.route('/api/add_message/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    
    time_ini = time.time()
    
    ### Verificar formato da entrada do texto
    content = request.json
    df = pd.json_normalize(content)
    
    ### Verificar se precisa passar o caminho aqui!!!
    bow_tfidf = nlp.use_TFIDF_Vec_model(df.texto.tolist())
        
    #y_pred = skill_model.predict(bow_tfidf)
    pred_skills = skill_model.predict_proba(bow_tfidf)[:, 1]
    
    #### Montar saída!!!
    # Print best n matches
    n=3
    best_n = np.argsort(pred_skills, axis=1)[:,-n:]
    classes = skill_model.classes_
    
    # TO DO >>> Loop the n options >> need to do !!!
    print(f'First predicted class = {classes[best_n[0, 2]]} and confidence = {pred_skills[0, best_n[0, 2]]:.2%}')
    print(f'Second predicted class = {classes[best_n[0, 1]]} and confidence = {pred_skills[0, best_n[0, 1]]:.2%}')
    print(f'Third predicted class = {classes[best_n[0, 0]]} and confidence = {pred_skills[0, best_n[0, 0]]:.2%}')
    
    time_end = time.time()
    print(f"Time 6: {time_end - time_ini} seconds")
    
    ### Verificar o que e como (formato) que vai retornar, se for um JSON, precisa dar um tapa
    
    return ??????

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)
    
    
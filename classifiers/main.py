# %%
import os
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

import dnd_5e_checks
from NLP_Classifier import NLP_Classifier

# %%
TARGET_VARIABLE = 'SKILL'
TEXT_VARIABLE = 'BACKWARD_TEXT'
PATH_DIR_MODELS = './models/'
PATH_DIR_DATA = '../parsers/'

# %%
classifier_model = dnd_5e_checks.Model_Trainer()
# %%
classifier_model.train_skill_classification(PATH_DIR_MODELS)

# %%
df_critical_role = pd.read_csv(
    PATH_DIR_DATA + 'critical_role/skills_dataset.csv',
    sep=',',
    error_bad_lines=False)

df_general_podcasts = pd.read_csv(
    PATH_DIR_DATA + 'podcasts/general_podcasts.csv',
    sep=',',
    error_bad_lines=False)

df_tavern_keeper = pd.read_csv(
    PATH_DIR_DATA + 'tavern_keeper/skills_dataset.csv',
    sep=',',
    error_bad_lines=False)

df_list = [
    df_critical_role,
    df_general_podcasts,
    df_tavern_keeper]
# %%
y_true_list = [ df.skill.to_list() for df in df_list ]

# %%
### Verificar se precisa passar o caminho aqui!!!
filename = os.path.join(PATH_DIR_MODELS, 'model.sav' )
skill_model = pickle.load(open(filename, 'rb'))

filename = os.path.join(PATH_DIR_MODELS, 'tfidf.pickle' )
skill_tfidf = pickle.load(open(filename, 'rb'))
nlp = NLP_Classifier()

# %%
bow_tfidf_list = [nlp.use_TFIDF_Vec_model(
    df.backward_text.tolist(), PATH_DIR_MODELS)
    for df in df_list]

# %%
predicted_probabilities_for_classes_list = [
    skill_model.predict_proba(bow_tfidf)
    for bow_tfidf in bow_tfidf_list
]
# %%
classes = skill_model.classes_
predicted_winner_index_list = [
    np.argmax(predicted_probabilities, axis=1)
    for predicted_probabilities in predicted_probabilities_for_classes_list]

# %%
predicted_classes_list = [
    [classes[index] for index in predicted_winner_indexes]
    for predicted_winner_indexes in predicted_winner_index_list]

# %%
for y_true, y_pred in zip(y_true_list, predicted_classes_list):
    print(f1_score(y_true, y_pred, average='micro'))

# %%
predicted_winner_class_and_confidence_list = [
    [(classes[np.argmax(predicted_row)], np.max(predicted_row))
        for predicted_row in predicted_probabilities]
    for predicted_probabilities in predicted_probabilities_for_classes_list]

# %%
score_list = []
for y_true_list, y_pred_and_confidence_list in zip(
        y_true_list, predicted_winner_class_and_confidence_list):
    score_column = []
    for y_true, (y_pred, confidence) in zip(
            y_true_list, y_pred_and_confidence_list):
        if y_true == y_pred:
            score_column.append(confidence)
        else:
            score_column.append(confidence * -1)
    score_list.append(score_column)
# %%
for df, score_column in zip(df_list, score_list):
    df['row_score'] = score_column

# %%
df_list[0].to_csv("../data/critical_role")
# %%
df_list[1].to_csv("../data/general_podcasts")

# %%
df_list[2].to_csv("../data/tavern_keeper")
# %%
dataframe = pd.concat(df_list)
mask = dataframe['row_score'] > 0
dataframe = dataframe[mask].info()
# %%
classifier_model.train_skill_classification(PATH_DIR_MODELS, dataframe)

# %%
dataframe = pd.concat(df_list)
mask = dataframe['row_score'] > -0.2
dataframe = dataframe[mask].info()
# %%
classifier_model.train_skill_classification(PATH_DIR_MODELS, dataframe)

# %%
dataframe = pd.concat(df_list)
mask = dataframe['row_score'] > 0.3
dataframe = dataframe[mask].info()

# %%
classifier_model.train_skill_classification(PATH_DIR_MODELS, dataframe)

# %%
dataframe = pd.concat(df_list)
mask = dataframe['row_score'] > 0.5
dataframe = dataframe[mask].info()

# %%
classifier_model.train_skill_classification(PATH_DIR_MODELS, dataframe)

# %%
classifier_model.lst_skills.append('NoCheck')
# %%
dataframe = pd.concat(df_list)
mask = dataframe['row_score'] < 0.3
dataframe.loc[mask, 'skill'] = 'NoCheck'
dataframe.info()
# %%
classifier_model.train_skill_classification(PATH_DIR_MODELS, dataframe)

# %%
dataframe = pd.concat(df_list[:2])
mask = dataframe['row_score'] < 0.3
dataframe.loc[mask, 'skill'] = 'NoCheck'
dataframe.info()

# %%
classifier_model.train_skill_classification(PATH_DIR_MODELS, dataframe)


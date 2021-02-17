import pandas as pd
import re
import spacy
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.util import ngrams

from unidecode import unidecode

CR_url = 'https://raw.githubusercontent.com/amiapmorais/datasets/master/critical_role/skills_dataset.csv'
TK_url = 'https://raw.githubusercontent.com/amiapmorais/datasets/master/tavern_keeper/skills_dataset.csv'
SS_url = 'https://raw.githubusercontent.com/amiapmorais/datasets/master/skill_db.csv'
GP_url = 'https://raw.githubusercontent.com/amiapmorais/datasets/master/podcasts/general_podcasts.csv'

fields = {'skill', 'backward_text'}

df_critical_role = pd.read_csv(CR_url, usecols=fields)
df_tavern_keeper_5e = pd.read_csv(TK_url, usecols=fields)
df_skill_sheet = pd.read_csv(SS_url, usecols=fields)
df_general_podcast = pd.read_csv(GP_url, usecols=fields)

# Flag data source
df_critical_role['origin'] = 'CR'
df_tavern_keeper_5e['origin'] = 'TK'
df_skill_sheet['origin'] = 'SS'
df_general_podcast['origin'] = 'GP'

"""
DEBUG - Alguma fonte de dados está zoada!!!
df_tavern_keeper_5e >> Está com dados zoados >> prejudica o modelo
df_DEBUG = df.groupby('skill').apply(pd.DataFrame.sample, n=5, replace=True).reset_index(drop=True)
"""
# Append all dataframes
#list_df = [df_critical_role, df_tavern_keeper_5e]
list_df = [df_critical_role, df_skill_sheet, df_general_podcast]
df = df_skill_sheet.append(list_df, ignore_index=True)

# Cleans text from processing and tokenizing
def strip_nonalpha(text):
    text = text.lower()
    t = unidecode(text)
    t.encode("ascii")  
    t = re.sub(r'[^a-z]', ' ', t)           #Remove nonalpha
    t = re.sub(r'\s[^a-z]\s', ' ', t)       #Remove nonalpha >> check if is really necessary!?!?
    t = re.sub(r"\b[a-z]{1,2}\b", ' ', t)   #Remove words with 1 or 2 letters
    t = re.sub(' +', ' ', t)                #Remove extra spaces
    t = t.strip()                           #Remove leading and trailing spaces
    return t

# Drop rows without text for model
df = df.dropna()

# Cleans text
df['clean_text'] = df['backward_text']

#df_DEBUG = df.groupby('skill').apply(pd.DataFrame.sample, n=5, replace=True).reset_index(drop=True)

nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer(language='english')

new_stopwords = {
                #Common words
                "going", "right", "okay", "yeah", "want", "try", "gonna", "good", "yes", "look", "know", "way", "looks", "guy",
                "little", "check", "thin", "thing", "guys", "come", "roll", "let", "time", "got", "goes", "maybe", "don", "think", 
                "let", "got",
                #PCs and NPCs Names
                "jester", "caleb", "nott", "fjord", "yasha", "beau", "matt", "sam", "travis", "marisha", "ashley", "laura", 
                "liam", "professor", "thaddeus", "taliesin", "mollymauk", "grog", "pike"
                }

stopwords = spacy.lang.en.stop_words.STOP_WORDS
stopwords.update(new_stopwords)

print(stopwords)

"""
Try options with lemma and stem
"""
def tokenize(str_text):
    doc = nlp(str_text)
    # Remove stop Words, keeps verbs and nouns
    tokens = [token.text for token in doc if (not token.is_stop) and (token.pos_ == 'VERB' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ')]

    return ' '.join(tokens)

def stemmize(str_text):
    doc = nlp(str_text)
    # Remove stop Words, keeps verbs and nouns
    tokens = [token.text for token in doc if (not token.is_stop) and (token.pos_ == 'VERB' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ')]
    stemms = [stemmer.stem(token) for token in tokens]

    return ' '.join(stemms)

def lemmanize(str_text):
    doc = nlp(str_text)
    # Remove stop Words, keeps verbs and nouns
    tokens = [token.lemma_ for token in doc if (not token.is_stop) and (token.pos_ == 'VERB' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ')]
    return ' '.join(tokens)

def ngramnizer(str_text, n):
    token = nltk.word_tokenize(str_text)
    token_grams = ngrams(token, n)
    return ' '.join(token_grams)

df['train_text'] = df['clean_text'].apply(tokenize)

df['stemm_text'] = df['clean_text'].apply(stemmize)

df['lemma_text'] = df['clean_text'].apply(lemmanize)

import os

def root_path():
    return os.path.abspath(os.sep)

def folder(*args):
    return os.path.join(root_path(), *args)

path_obs = folder('app', 'rpgai', 'data', 'rpgai_text.parquet')

df.to_parquet(path_obs, index=False)

#df['bigrams_text'] = df['lemma_text'].apply(ngramnizer, args=(2,))

# Check data distribution per skill
df.skill.value_counts()

#Retira skills não mapeados
not_skill = ['disguise', 'concentration']
df = df[~df.skill.isin(not_skill)]

# Do an oversampling to try to get better prediction
df_estrat = df.groupby('skill').apply(pd.DataFrame.sample, n=400, replace=True).reset_index(drop=True)

# Model Train and Selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np

# Bag of words
#count_vect = CountVectorizer(max_features=None, min_df=3, ngram_range=(2, 2))
count_vect = CountVectorizer()
bow = count_vect.fit_transform(df_estrat['stemm_text'])
# tf-idf
tfidf_transformer = TfidfTransformer()
bow_tfidf = tfidf_transformer.fit_transform(bow)

# split data for train and test
X_train, X_test, y_train, y_test = train_test_split(bow_tfidf, df_estrat['skill'], test_size=0.2, random_state = 42)

# Train Model
#clf = LinearSVC()
#clf = XGBClassifier(objective = 'binary:logistic')
#clf = SVC() 
#clf.probability=True
clf = RandomForestClassifier(n_estimators=200)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {metrics.precision_score(y_test, y_pred, average='macro'):.2%}")

"""
Check some cases to analyze the model
"""
acrobatics = "you tumble the strike"
animal = "No it is just a regular sized dog that has three heads and is like all blue and ethereal. And they get right up in each others faces, and then just growls at each other, [growling] and both of them are in a dead stop. They are locked eyes at each other."
athletics = "So, I'm going to reach up and use the Long Arm of the Law to kind of grapple the edge of the doorway, and then kind of like ferry people up and climb up my body and my arm to the doorway."
arcana = "Hey, am I passively sensing any kind of magic?"
deception = "We want to look like nondescript peasants. So if somebody saw us there, they would assume we were there to clean, to deliver food, to do whatever needs doing in the house."
history = "I guess I haven't seen it in a really long time I don’t know where it is. I am looking for this uh mouthpiece, I don’t know what it’s attached to now. But it looks like a very large open mouth and it’s laughing, and it has really bright red lips and I don’t know if you’ve seen it before, but if you see it I would love to get my hand on it cause I think it’s tied to all these things happening right now."
insight = "Like a suggestive wink or like we're on the same team wink? Can I investigate the wink?"
intimidation = "I am going to push my cloak aside to have one hand and my dagger and hold my hand up knowing that I cannot stop him with the body force but I can stop him with intimidation and say"
investigation = "I’d love to search the ceiling to see if there are any hatches."
medicine = "Johnny, do you know anything about this frozen poison type thing? Can you help him right now?"
nature = "I think back on my adventures in nature to perhaps recognize what they are."
perception = "Do I notice anything dangerous?"
performance = "And pulls out his help horn, runs across the room while blowing it and then pulls his maroon cape in front of him as if he's egging Nessie on."
persuasion = "But when this is all over and we trap the Council, and everything goes back to normal, I don’t want anyone to know who I am. I want a fresh start. Except for the friends and the family I made. Inara should know, and everyone else, Evan. But the world at large, I don’t want any fame, fortune, I just want to go. I don’t want anyone to know I’m the reason this started, this whole thing started. I just want to go sell tea."
religion = "I would like my religious senses to let me know any kind of divine interference that might be going on."
sleight = "Before Bridge walks away, Inara’s going to pickpocket him for gold."
stealth = "Come out that side and then get up into the ring, behind the barrel as cover, and then attack the guy that way."
survival = "The atmosphere is breaking his concentration on his main task of checking around the party for travelling advantages...!"

def check_for_skill(skill_name, skill, n):
    skill_test = stemmize(strip_nonalpha(skill))
    y_valid = clf.predict(count_vect.transform([skill_test]))
    y_valid_prob = clf.predict_proba(count_vect.transform([skill_test]))
    
    # Print best n matches
    best_n = np.argsort(y_valid_prob, axis=1)[:,-n:]
    classes = clf.classes_
    print(f'Expected: {skill_name}')
    print(f'Predicted: {y_valid[0]}')
    
    # TO DO >>> Loop the n options >> need to do !!!
    print(f'First predicted class = {classes[best_n[0, 2]]} and confidence = {y_valid_prob[0, best_n[0, 2]]:.2%}')
    print(f'Second predicted class = {classes[best_n[0, 1]]} and confidence = {y_valid_prob[0, best_n[0, 1]]:.2%}')
    print(f'Third predicted class = {classes[best_n[0, 0]]} and confidence = {y_valid_prob[0, best_n[0, 0]]:.2%}')
    return y_valid

# Skill Check to validate
skill_to_check = check_for_skill('arcana', arcana, 3)

"""
"""
def skills():

    skills = [
    'Deception',
    'Intimidation',
    'Performance',
    'Persuasion',
    'Acrobatics',
    'Sleight of Hand',
    'Stealth',
    'Arcana',
    'History',
    'Investigation',
    'Nature',
    'Religion',
    'Athletics',
    'Animal Handling',
    'Insight',
    'Medicine',
    'Perception',
    'Survival'
    ]

    return skills


import os
import sys
sys.path.append(os.getcwd())

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# The data frame needs the skill and train_text columns
def wordcloud_by_df(df):
  
  for skill in skills():
    df_train_text = df[df['skill'] == skill]
    document = ' '.join(df_train_text['stemm_text'])
    wordcloud(document)

def wordcloud(text):
  wordcloud = WordCloud(background_color="white").generate(text)
  plt.figure(figsize=(10,10))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")


wordcloud_by_df(df_estrat)
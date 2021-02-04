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

fields = {'skill', 'backward_text'}

df_critical_role = pd.read_csv(CR_url, usecols=fields)
df_tavern_keeper_5e = pd.read_csv(TK_url, usecols=fields)
df_skill_sheet = pd.read_csv(SS_url, usecols=fields)

# Flag data source
df_critical_role['origin'] = 'CR'
df_tavern_keeper_5e['origin'] = 'TK'
df_skill_sheet['origin'] = 'SS'

"""
DEBUG - Alguma fonte de dados está zoada!!!
df_tavern_keeper_5e >> Está com dados zoados >> prejudica o modelo
df_DEBUG = df.groupby('skill').apply(pd.DataFrame.sample, n=5, replace=True).reset_index(drop=True)
"""
# Append all dataframes
#list_df = [df_critical_role, df_tavern_keeper_5e]
list_df = [df_critical_role]
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
df['clean_text'] = df['backward_text'].apply(strip_nonalpha)

df_DEBUG = df.groupby('skill').apply(pd.DataFrame.sample, n=5, replace=True).reset_index(drop=True)

nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer(language='english')

new_stopwords = {
                #Common words
                "going", "right", "okay", "yeah", "want", "try", "gonna", "good", "yes", "look", "know", "way", "looks", "guy",
                "little", "check", "thin", "thing", "guys", "come", "roll", "let", "time", "got", "goes", "maybe", 
                #PCs and NPCs Names
                "jester", "caleb", "nott", "fjord", "yasha", "beau", "matt", "sam", "travis", "marisha", "ashley", "laura", 
                "liam", "professor", "thaddeus", "taliesin", "mollymauk", "grog", "pike"
                }

stopwords = spacy.lang.en.stop_words.STOP_WORDS
stopwords.update(new_stopwords)

"""
Try options with lemma and stem
"""
def tokenize(str_text):
    doc = nlp(str_text)
    # Remove stop Words, keeps verbs and nouns
    tokens = [token.text for token in doc if (not token.is_stop) and token.pos_ == 'VERB' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ']
    return ' '.join(tokens)

def stemmize(str_text):
    doc = nlp(str_text)
    # Remove stop Words, keeps verbs and nouns
    tokens = [token.text for token in doc if (not token.is_stop) and token.pos_ == 'VERB' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ']
    stemms = [stemmer.stem(token) for token in tokens]

    return ' '.join(stemms)

def lemmanize(str_text):
    doc = nlp(str_text)
    # Remove stop Words, keeps verbs and nouns
    tokens = [token.lemma_ for token in doc if (not token.is_stop) and token.pos_ == 'VERB' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ']
    return ' '.join(tokens)

def ngramnizer(str_text, n):
    token = nltk.word_tokenize(str_text)
    token_grams = ngrams(token, n)
    return ' '.join(token_grams)


df['train_text'] = df['clean_text'].apply(tokenize)

df['stemm_text'] = df['clean_text'].apply(stemmize)

df['lemma_text'] = df['clean_text'].apply(lemmanize)

#df['bigrams_text'] = df['clean_text'].apply(ngramnizer, args=(2,))

# Check data distribution per skill
df.skill.value_counts()

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
count_vect = CountVectorizer()
bow = count_vect.fit_transform(df_estrat['lemma_text'])
# tf-idf
tfidf_transformer = TfidfTransformer()
bow_tfidf = tfidf_transformer.fit_transform(bow)

# split data for train and test
X_train, X_test, y_train, y_test = train_test_split(bow_tfidf, df_estrat['skill'], test_size=0.25, random_state = 42)

# Train Model
#clf = LinearSVC()
#clf = XGBClassifier(objective = 'binary:logistic')
#clf = SVC() 
#clf.probability=True
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {metrics.precision_score(y_test, y_pred, average='macro'):.2%}")

"""
Check some cases to analyze the model
"""
acrobatics = 'seeing his princess wrap a black cord he says, oh this is gonna be fun, much appreciated.  "Metal, second verse same as the first " as he tumbles behind the next living spell '
athletics = 'ASHLEY: Right, okay. MATT: That finishes its turn. Beau, you are up. You watch Yasha slam on the ground, unconscious next to you, the blade clattering to the ground and coming to rest. The creature lifts up (wheezing) and vanishes into the stone above you. MARISHA: I can not get a reaction from it, as it goes? MATT: It was not close enough to you, unfortunately. MARISHA: Fuck. Im going to run over to this bookcase and put my staff behind it to see if I can knock it over. MATT: You get the staff on the fulcrum.'
survival = 'Thanks to Halbarad s advice and map, Ren felt prepared for the route they would take on the journey.'
insight = 'Will pay keen attention to read into any suggestion of how the news is presented to Thorin  and how welcome it is to him. Zaken has a sneaky feeling that Thorin has something lingering in his mind  from the meeting they had yesterday anyway.'
religion = 'i try recognize the holy symbol'
acrobatics2 = 'you tumble the strike'

def check_for_skill(skill_name, skill, n):
    skill_test = lemmanize(strip_nonalpha(skill))
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
skill_to_check = check_for_skill('acrobatics2', acrobatics2, 3)




"""

TO DO >> Improve the data analysis


"""

def BOW(skill_name):
  skills_backward = df[df['skill'] == skill_name]
  backward_train, backward_test, skill_train, skill_test = train_test_split(skills_backward['train_text'], skills_backward['skill'], random_state = 0)
  
  count_vect_skill = CountVectorizer()
  count_vect_skill.fit_transform(backward_train)
  print(count_vect_skill.get_feature_names())
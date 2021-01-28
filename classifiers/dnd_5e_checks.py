import pandas as pd

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

df_critical_role = pd.read_csv('https://raw.githubusercontent.com/amiapmorais/datasets/master/critical_role/skills_dataset.txt', sep=';', error_bad_lines=False)
df_tavern_keeper = pd.read_csv('https://raw.githubusercontent.com/amiapmorais/datasets/master/tavern_keeper/skills_dataset.csv')

# Filtrando o dataset do Tavern Keeper apenas pelas skills 5e e salvando em uma cópia
df_tavern_keeper_5e = df_tavern_keeper[df_tavern_keeper['skill'].isin(skills)].copy()


# Seleciona origem do treinamento
df_critical_role['origin'] = 'CR'
df_tavern_keeper_5e['origin'] = 'TK'

# Juntando em um unico data frame
list_df = [df_critical_role, df_tavern_keeper_5e]
df = pd.DataFrame(columns=['skill', 'check_line', 'backward_text', 'origin'])
df = df.append(list_df, ignore_index=True)

# Sem essa conversão, acontece um erro na hora de remover as stop words
df['backward_text'] = df['backward_text'].astype(str)
df['backward_text'] = df['backward_text'].str.lower()

import spacy

nlp = spacy.load("en_core_web_sm")

# Remove palavras muito comuns
nlp.vocab["going"].is_stop = True
nlp.vocab["right"].is_stop = True
nlp.vocab["okay"].is_stop = True
nlp.vocab["yeah"].is_stop = True
nlp.vocab["want"].is_stop = True
nlp.vocab["try"].is_stop = True
nlp.vocab["gonna"].is_stop = True
nlp.vocab["good"].is_stop = True
nlp.vocab["yes"].is_stop = True
nlp.vocab["no"].is_stop = True
nlp.vocab["oh"].is_stop = True
nlp.vocab["look"].is_stop = True
nlp.vocab["know"].is_stop = True
nlp.vocab["way"].is_stop = True
nlp.vocab["looks"].is_stop = True
nlp.vocab["guy"].is_stop = True
nlp.vocab["little"].is_stop = True
nlp.vocab["check"].is_stop = True
nlp.vocab["thin"].is_stop = True
nlp.vocab["thing"].is_stop = True
nlp.vocab["guys"].is_stop = True
nlp.vocab["come"].is_stop = True
nlp.vocab["roll"].is_stop = True
nlp.vocab["let"].is_stop = True
nlp.vocab["time"].is_stop = True
nlp.vocab["got"].is_stop = True
nlp.vocab["goes"].is_stop = True
nlp.vocab["maybe"].is_stop = True

# Remove nome dos players e personagens
nlp.vocab["jester"].is_stop = True
nlp.vocab["caleb"].is_stop = True
nlp.vocab["nott"].is_stop = True
nlp.vocab["fjord"].is_stop = True
nlp.vocab["yasha"].is_stop = True
nlp.vocab["beau"].is_stop = True
nlp.vocab["matt"].is_stop = True
nlp.vocab["sam"].is_stop = True
nlp.vocab["travis"].is_stop = True
nlp.vocab["marisha"].is_stop = True
nlp.vocab["ashley"].is_stop = True
nlp.vocab["laura"].is_stop = True
nlp.vocab["liam"].is_stop = True
nlp.vocab["professor"].is_stop = True
nlp.vocab["thaddeus"].is_stop = True
nlp.vocab["taliesin"].is_stop = True
nlp.vocab["mollymauk"].is_stop = True
nlp.vocab["grog"].is_stop = True
nlp.vocab["pike"].is_stop = True

import re

def tokenize(str_text):

  # Manter apenas palavras
  words = re.sub(r"[^a-z]", ' ', str_text)
  # Remove palavras menores que 2 letras
  words = re.sub(r"\b[a-z]{1,2}\b", ' ', words)

  doc = nlp(words)
  # Remove stop Words, pontuação, mantendo apenas verbos e substantivos
  tokens = [token.text for token in doc if (not token.is_stop | token.is_punct) and token.pos_ == 'VERB' or token.pos_ == 'NOUN']
  
  return ' '.join(tokens)


df['train_text'] = df['backward_text'].apply(tokenize)


# Amostra estratificada com reposição >> Risco de viciar o modelo
df_estrat = df.groupby('skill').apply(pd.DataFrame.sample, n=500, replace=True).reset_index(drop=True)

# Descomentar quando a massa de dados tiver mais que 300 exemplos de cada skill
#df_estrat = df.groupby('skill').apply(pd.DataFrame.sample, n=300).reset_index(drop=True)

#Treinando o modelo ------------------------

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Split dos dados em treino e validação
X_train, X_test, y_train, y_test = train_test_split(df_estrat['train_text'], df_estrat['skill'], random_state = 0)

#Adicionando as ações da planilha a base de treinamento
df_skill_sheet = pd.read_csv('https://raw.githubusercontent.com/amiapmorais/datasets/master/skill_db.csv')
df_skill_sheet['origin'] = 'SS'

df_skill_sheet['backward_text'] = df_skill_sheet['backward_text'].astype(str)
df_skill_sheet['backward_text'] = df_skill_sheet['backward_text'].str.lower()

df_skill_sheet['train_text'] = df['backward_text'].apply(tokenize)

X_train = X_train.append(df_skill_sheet['train_text'])
y_train = y_train.append(df_skill_sheet['skill'])

# Bag of words
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# Transformando o bag of words em um tf-idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Treinando o modelo
clf = LinearSVC().fit(X_train_tfidf, y_train)

y_pred = clf.predict(count_vect.transform(X_test))

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def print_wordcloud(skill_name):
  skills_backward = df[df['skill'] == skill_name]
  document = ' '.join(skills_backward['train_text'])

  wordcloud = WordCloud(background_color="white").generate(document)
  plt.figure(figsize=(10,10))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  print(skill_name)

def BOW(skill_name):
  skills_backward = df[df['skill'] == skill_name]
  backward_train, backward_test, skill_train, skill_test = train_test_split(skills_backward['train_text'], skills_backward['skill'], random_state = 0)
  
  count_vect_skill = CountVectorizer()
  count_vect_skill.fit_transform(backward_train)
  print(count_vect_skill.get_feature_names())
# -*- coding: utf-8 -*-
"""classifier-dnd-checks.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NWguoLlvOZz1bPRUor4oBCS8XJrcp2A-

# Obter os dados
"""

import pandas as pd

#df = pd.read_csv(datasets_path.joinpath('skills_dataset.txt'), sep=';', error_bad_lines=False)
df = pd.read_csv('https://raw.githubusercontent.com/amiapmorais/rpgai/main/skills_dataset.txt', sep=';', error_bad_lines=False)
df.head()

df.sample().values[0]

# Remove a percepção porque tinha mais de 1300
#df = df[df.skill != 'Perception']

#Apenas com dados do critical role
df.groupby('skill').count()

import plotly.graph_objects as go
grafico_label = go.Figure()
grafico_label.add_trace(go.Histogram(histfunc="count",  x=df['skill']))
grafico_label

"""## Obtendo as ações da planilha"""

df_actions = pd.read_csv('https://raw.githubusercontent.com/amiapmorais/rpgai/main/SkillDB-csv.csv', error_bad_lines=False)
df_actions.head()

# Seleciona origem do treinamento
df['origem'] = 'CR'
df_actions['origem'] = 'RR'
df = df.append(df_actions, ignore_index=True)

df_actions.groupby('skill').count()

"""# Limpar os dados"""

import spacy

# Sem essa conversão, acontece um erro na hora de remover as stop words
df['backward_text'] = df['backward_text'].astype(str)
df['backward_text'] = df['backward_text'].str.lower()

nlp = spacy.load("en_core_web_sm")

# remove palavras muito comuns
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


# remove nome dos players e personagens
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

#Removendo as stop words, pontuação e números. Depois concatenando tudo junto
for i,backward_text in df.iterrows():
  doc = nlp(backward_text['backward_text'])
  #Tokenize. cria um array com o texto dos tokens que não são Stop Words, pontuação ou números
  tokens = [token.text for token in doc if not token.is_stop | token.is_punct| token.text.isdigit()]

  df.at[i,'backward_text'] = ' '.join(tokens)

import re

def clean_text(text):
  
  text_return = re.sub(r"[^a-z]", ' ', text) # keep only words
  text_return = re.sub(r"\b[a-z]{1,2}\b", ' ', text_return) # remove <2 characters
  return text_return

df['backward_text'] = df['backward_text'].apply(clean_text)

df

# Amostra estratificada com reposição >> Risco de viciar o modelo
df_estrat = df.groupby('skill').apply(pd.DataFrame.sample, n=300, replace=True).reset_index(drop=True)

# Descomentar quando a massa de dados tiver mais que 300 exemplos de cada skill
#df_estrat = df.groupby('skill').apply(pd.DataFrame.sample, n=300).reset_index(drop=True)

df.sample().values[0]

# Naive Bayes Classifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

"""X_train: 75% dos textos prévios ao teste
X_test: 25% dos labels da skills
y_train: 75% dos labels das skills
y_test: 25% dos labels das skills
"""

# Split dos dados em treino e validação
X_train, X_test, y_train, y_test = train_test_split(df_estrat['backward_text'], df_estrat['skill'], random_state = 0)

#Adicionando as ações da planilha a base de treinamento
#X_train = X_train.append(df_actions['backward_text'])
#y_train = y_train.append(df_actions['skill'])

# Bag of words
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

"""X_train_counts é uma matriz que representa quantas vezes cada uma das palavras apareceram no corpus, onde a quantidade de linhas do corpus é a quantidade de linhas da matriz e a quantidade de colunas é o tamanho do bag of words, e cada celula é a quantidade de vezes que a palavra que está na mesma posição do bag of words apareceu na linha do corpus em questão"""

#ta muito zuado as features, não devia ter números
print(count_vect.get_feature_names())

# Transformando o bag of words em um tf-idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Commented out IPython magic to ensure Python compatibility.
# Treinando o modelo
# pra treinar o modelo usa os 75% dos textos e dos labels, pra depois ele predizer
#clf = MultinomialNB().fit(X_train_tfidf, y_train)
# %time clf = LinearSVC().fit(X_train_tfidf, y_train)

#aqui ele cria um array do que foi predito dos 25% de teste de treinamento que não foi usado pra treinar o modelo
y_pred = clf.predict(count_vect.transform(X_test))

print(y_pred)

from sklearn import metrics
#Adicionando as ações da planilha a acuracia foi de 0.45254901960784316 para 0.47058823529411764
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#aqui ele pega a precisão de cara feature, porque ele compara a feature o acerto de cada feature ordenada do menor para o maior (em caso de números), ou ordem alfabética
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))

acrobatics = 'seeing his princess wrap a black cord he says, oh this is gonna be fun, much appreciated.  "Metal, second verse same as the first " as he tumbles behind the next living spell '
athletics = 'ASHLEY: Right, okay. MATT: That finishes its turn. Beau, you are up. You watch Yasha slam on the ground, unconscious next to you, the blade clattering to the ground and coming to rest. The creature lifts up (wheezing) and vanishes into the stone above you. MARISHA: I can not get a reaction from it, as it goes? MATT: It was not close enough to you, unfortunately. MARISHA: Fuck. Im going to run over to this bookcase and put my staff behind it to see if I can knock it over. MATT: You get the staff on the fulcrum.'
survival = 'Thanks to Halbarad s advice and map, Ren felt prepared for the route they would take on the journey.'
insight = 'Will pay keen attention to read into any suggestion of how the news is presented to Thorin  and how welcome it is to him. Zaken has a sneaky feeling that Thorin has something lingering in his mind  from the meeting they had yesterday anyway.'
religion = 'i try recognize the holy symbol'
acrobatics2 = 'you tumble the strike'

# ao invés de dar o predict, retornar os 3 maiores skills com probabilidade prevista
print(clf.predict(count_vect.transform([acrobatics2.lower()])))

"""# Matriz de confusão"""

import numpy as np
skill_arr = np.array(df['skill'].unique())
skill_arr = np.sort(skill_arr)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=skill_arr, yticklabels=skill_arr)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

y_test.value_counts()

"""# Wordclouds + bag of words"""

!pip install wordcloud

from wordcloud import WordCloud

# Fazer o wordcloud com base no df_estrat ao invés de df

def print_wordcloud(skill_name):
  skills_backward = df[df['skill'] == skill_name]
  document = ' '.join(skills_backward['backward_text'])

  wordcloud = WordCloud(background_color="white").generate(document)
  plt.figure(figsize=(10,10))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  print(skill_name)

def BOW(skill_name):
  skills_backward = df[df['skill'] == skill_name]
  backward_train, backward_test, skill_train, skill_test = train_test_split(skills_backward['backward_text'], skills_backward['skill'], random_state = 0)
  
  count_vect_skill = CountVectorizer()
  count_vect_skill.fit_transform(backward_train)
  print(count_vect_skill.get_feature_names())

print_wordcloud('Deception')

BOW('Deception')

print_wordcloud('Intimidation')

BOW('Intimidation')

print_wordcloud('Performance')

BOW('Performance')

print_wordcloud('Persuasion')

BOW('Persuasion')

print_wordcloud('Acrobatics')

BOW('Acrobatics')

print_wordcloud('Sleight of Hand')

BOW('Sleight of Hand')

print_wordcloud('Stealth')

BOW('Stealth')

print_wordcloud('Arcana')

BOW('Arcana')

print_wordcloud('History')

BOW('History')

print_wordcloud('Investigation')

BOW('Investigation')

print_wordcloud('Nature')

BOW('Nature')

print_wordcloud('Religion')

BOW('Religion')

print_wordcloud('Athletics')

BOW('Athletics')

print_wordcloud('Animal Handling')

BOW('Animal Handling')

print_wordcloud('Insight')

BOW('Insight')

print_wordcloud('Medicine')

BOW('Medicine')

print_wordcloud('Perception')

BOW('Perception')

print_wordcloud('Survival')

BOW('Survival')

"""#TODO

1.   Fazer bag of words para cada Skill/wordcloud (antes de diminuir o número de linhas) **FEITO**
2.   Printar a matriz de confusão: heatmap do previsto x realizado **FEITO**
3.   Usar o steamming do Spacy, basicamente usar o radical das palavras, retirando plurais e conjugações.
1.   Treinar o classificador adicionando os textos da planilha SKILL DB / Testar a acurácia com eles **FEITO** 
1.   Reduzir o número de linhas usados como backward_text para verificar se melhora as palavras
4.   Usar o Spacy para filtrar apenas verbos e substantivos
5.   Com o bag of words tentar diferentes percentuais acumulados para construir as features com as palavras mais comuns para cada skill. Faz a limpeza do ruído e mantem apenas o sinal.

TO DOs
Lista de atividades pendentes para refinar o modelo e o pipeline, sem uma ordem específica:

Abrigar o código no git hub e abrir esses pontos de melhoria como issues que vamos baixando utilizando pull requests

Testar outros tipos de modelo (decision tree, random forest e NN - Neural Networks costumam ter um bom desempenho neste tipo de problema)

Ajustar a chamada da predição para trabalhar com a string com os mesmos tratamentos utilizados no treino (deve melhorar a eficiência)

Aumentar a quantidade de dados para treino

Separar o pipeline em scrapping (por fonte de dados), data prep (por fonte de dados), enrichment (combinação das fontes de dados), treino do modelo, escoragem e validação do modelo

Verificar uma forma mais elegante de passar / criar stopwords em grande volume no Spacy

Fazer análise de wordcloud utilizando os dados de treinamento

Fazer análise de bag of words utilizando os dados de treinamento

Fazer análise de bag of words com percentil acumulado de ocorrências (separar ruído de sinal) nas palavras que irão para o treinamento, colocar um ponto de corte

Testar modelo com menos linhas de texto do transcript, analisar se melhora o modelo.

Testar pegar apenas verbos e substantivos (POS - Part of Speech) com o Spacy, analisar se o modelo melhora

Testar lemmatização utilizando o Spacy, analisar se o modelo melhora.

Testar iterar pela parse tree do Spacy, testar se os ngramas (combinação de palavras que aparecem juntas) melhoram o modelo.

Extração de NER (nomes de players) utilizando o Spacy

ref. https://spacy.io/usage/linguistic-features
"""
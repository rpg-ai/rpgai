
import os
import sys
sys.path.append(os.getcwd())
from skills_5e import skills

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# The data frame needs the skill and train_text columns
def wordcloud_by_df(df):
  
  for skill in skills():
    df_train_text = df[df['skill'] == skill]
    document = ' '.join(df_train_text['train_text'])
    wordcloud(document)

def wordcloud(text):
  wordcloud = WordCloud(background_color="white").generate(text)
  plt.figure(figsize=(10,10))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
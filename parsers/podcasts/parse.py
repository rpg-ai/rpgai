import re
from pathlib import Path
import os
import pandas as pd
import sys
sys.path.append("..\\..\\utils")
from skills_5e import skills

NUMBER_OF_WORDS = 20

"""
Returns the regex pattern to find the checks
Checks example: 
  Roll me a deception 
  Give me an acrobatics roll
  Everybody give me a perception check real quick 
"""
def pattern(skill):
  return re.compile("roll(.*?)(?:"+ skill +")|" + skill + " (?:check|roll)", re.IGNORECASE)

def get_train_text(messages, check_position):
  text_position = check_position-1
  train_text = messages[text_position]
  #precisa arrumar isso aqui porque ele ta fazendo recursivamente até encontrar uma unica mensagem com mais de 20 palavras
  if len(train_text.split()) < NUMBER_OF_WORDS:
    # Recursivily concat the backward test
    return get_train_text(messages, text_position) + train_text

  else:
    return train_text

# Current Working Directory
cwd = Path(os.getcwd())
data_path = Path(cwd.joinpath('scraped_data/'))

players_char_names = ['Kyle', 'Ali', 'Spurrier', 'Goodrich', 'Yashee', 'Randy', "Raz’ul"]


skill_train_text = []

for text_file in data_path.iterdir():

  df = pd.read_csv(text_file)
  
  for session_transcript in df['selection1_transcript']:

    messages = session_transcript.split(':')

    for idx, message_text in enumerate(messages):

      for skill in skills():

        for match in re.finditer(pattern(skill), message_text):
          #print(message_text + '\n')
          dict_skill = {}
          dict_skill.update({'skill': skill, 'backward_text': get_train_text(messages, idx), 'check_line': message_text}) 
          skill_train_text.append(dict_skill)

    #esse break eh apenas para teste pra manter no primeiro arquivo
    break

df = pd.DataFrame(skill_train_text)
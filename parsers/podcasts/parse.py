import re
from pathlib import Path
import os
import pandas as pd
import sys
sys.path.append("..\\..\\utils")
from skills import skills_5e, skills_3_5e

df_checks = pd.read_csv('https://raw.githubusercontent.com/amiapmorais/datasets/master/5e_skills_dict.csv')

NUMBER_OF_WORDS = 30

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

  text_position = check_position - 1
  train_text = messages[text_position]
  
  if len(train_text.split()) < NUMBER_OF_WORDS:
    # Recursivily concat the backward test
    return get_train_text(messages, text_position) + train_text

  else:
    return train_text

# Remove o nome dos personagens
def remove_char_names(text):

  # bombarded
  bombard_names = ['Kyle', 'Ali', 'Spurrier', 'Goodrich', 'Yashee', 'Randy', "Raz’ul"]
  # adventurezone
  az_players_names = ['Griffin', 'Justin', 'Travis', 'Clint']
  az_char_names = ['Duck', 'Aubrey', 'Edmund', 'Taako', 'Merle', 'Magnus', 'Irene', 'Nadiya', 'Chris'
    'Augustus', 'Errol', 'Gandy']
  #encounter_party
  ep_players = ['Landree', 'Sarah', 'Ned', 'Brian']
  #gp
  gp_players = ['Lucifer', 'Lilith', 'Amanda', 'Syd', 'Cassidy']
  #join_party
  jp_players = ['Eric', 'Brandon', 'Lauren', 'Briggon', 'Julia']
  #magpie
  magpie_players = ['RHI', 'MADGE', 'JOSIE', 'KIM', 'MINNA']
  #alba
  alba_names = ['ZANE', 'GUBBIN', 'ROSINE', 'STAE', 'GUBBIE', 'BETULE', 'SEA', 'MIK', 'MIKE', 'SEAN', 'CARTER', 'ALB', 'MAGNU',
    'HOLL']

  char_names = bombard_names + az_players_names + ep_players + gp_players + jp_players + magpie_players + alba_names

  for name in char_names: 
    text = text.replace(name, ' ') 
  
  return " ".join(text.split())

# Retorna o nome da skill no DnD 5e com base no dicionário
def get_5e_skill_name(old_skill_name): 
  
  # Manter apenas palavras
  old_skill_name = re.sub(r"[^a-z]", ' ', old_skill_name.lower())
  
  df_skill = df_checks.loc[df_checks['text_skill'].str.contains(re.compile(r'\b({0})\b'.format(old_skill_name), flags=re.IGNORECASE))]
  return old_skill_name if df_skill.empty else df_skill.iloc[0]['5e_skill']

# Current Working Directory
cwd = Path(os.getcwd())
data_path = Path(cwd.joinpath('scraped_data/'))

skill_list = skills_5e() + skills_3_5e()

skill_train_text = []

for text_file in data_path.iterdir():

  df = pd.read_csv(text_file)

   # Conversão pra evitar erros de tipos de dados
  df['selection1_transcript'] = df['selection1_transcript'].astype(str)
  
  for session_transcript in df['selection1_transcript']:

    session_transcript = remove_char_names(session_transcript)

    messages = session_transcript.split(':')

    for idx, message_text in enumerate(messages):

      for skill in skill_list:

        for match in re.finditer(pattern(skill), message_text):

          train_text = get_train_text(messages, idx)
          dict_skill = {}
          dict_skill.update({
            'skill': get_5e_skill_name(skill), 
            'backward_text': ' '.join(train_text.split()[-NUMBER_OF_WORDS:]), 
            'check_line': message_text,
            'original_name': skill}) 
          
          skill_train_text.append(dict_skill)

df = pd.DataFrame(skill_train_text)
df.to_csv ('general_podcasts.csv', index = False, header=True)
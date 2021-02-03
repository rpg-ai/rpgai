import pandas as pd

# Ex de uma sessão: https://www.tavern-keeper.com/roleplay/10401

df_checks = pd.read_csv('https://raw.githubusercontent.com/amiapmorais/datasets/master/5e_skills_dict.csv')

# fonte do código: https://stackoverflow.com/questions/8153823/how-to-fix-this-attributeerror
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        # initialize the base class
        HTMLParser.__init__(self)

    def read(self, data):
        # clear the current output before re-use
        self._lines = []
        # re-set the parser's state before re-use
        self.reset()
        self.feed(data)
        return ''.join(self._lines)

    def handle_data(self, d):
        self._lines.append(d)

def strip_tags(html):
    s = MLStripper()
    return s.read(html)


import re

# Remove o nome dos personagens
def remove_char_names(text):
  char_names = set(re.findall('<div class="char-name">(.*?)</div>', text, re.DOTALL))
  
  for name in char_names: 
    text = text.replace(name, ' ') 
  
  return " ".join(text.split()) 

# Retorna o nome da skill no DnD 5e com base no dicionário
def get_5e_skill_name(old_skill_name): 
  
  # Manter apenas palavras
  old_skill_name = re.sub(r"[^a-z]", ' ', old_skill_name.lower())
  
  """
    Using a regex pattern in contains to get a whole world, not a sequence of characters
    The Craft check was return Arcana because 'Spellcraft' contains 'craft' before this modification
  """
  df_skill = df_checks.loc[df_checks['text_skill'].str.contains(re.compile(r'\b({0})\b'.format(old_skill_name), flags=re.IGNORECASE))]
  return old_skill_name if df_skill.empty else df_skill.iloc[0]['5e_skill']


from pathlib import Path
import os

# Current Working Directory
cwd = Path(os.getcwd())
data_path = Path(cwd.joinpath('scraped_data/'))

# Passo 1: Criando o dataframe com o texto das própias mensagens onde acontece a rolagem

skill_list = []
counter_skip = 0

# pegando o texto das mensagens onde tem as rolagens
for tk_csv_file in data_path.iterdir():
  
  df = pd.read_csv(tk_csv_file)
  # Conversão pra evitar erros de tipos de dados
  df['list1_page'] = df['list1_page'].astype(str)

  for html_page in df['list1_page']:
    # Removendo os nomes dos personagens
    html_page = remove_char_names(html_page)

    matches = re.findall('<div class="message-content">(.*?)<div class="msg-container face front">', html_page, re.DOTALL)
    for match in matches:
      if '<div class="dice-roll-block">' in match:
        
        train_text = re.findall('<p(.*?)<div class="dice-roll-block">', match, re.DOTALL)

        # Para evitar erros quando não encontra o pattern
        if not train_text:
          counter_skip += 1
          continue

        train_text = strip_tags(train_text[0]).replace('\n', '')
        train_text = train_text.split('>')[1]
        skills = re.findall('<b>(.*?)</b>', match, re.DOTALL)

        # Para pegar os testes duplos. tinha um total de 615 testes duplos
        for skill in skills:
          
          # Melhor processamento pra criar um dataframe é usando dict https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe/17496530#17496530
          dict_skill = {}
          dict_skill.update({'skill': get_5e_skill_name(skill.strip()), 'backward_text': train_text, 'original_name': skill.strip()}) 
          skill_list.append(dict_skill)
        
        #print('Found %s : %s' % (get_5e_skill_name(skill), train_text) )

df_message_roll = pd.DataFrame(skill_list)


# Passo 2: Criando um segundo dataframe usando o regex que encontra os testes de perícia

skill_list2 = []

for tk_csv_file in data_path.iterdir():

  df2 = pd.read_csv(tk_csv_file)
  # Conversão pra evitar erros de tipos de dados
  df2['list1_page'] = df2['list1_page'].astype(str)

  for html_page in df2['list1_page']:
    # Removendo os nomes dos personagens
    html_page = remove_char_names(html_page)

    matches = re.findall('<div class="message-content">(.*?)</div>', html_page, re.DOTALL)
    for match in matches:

      if '<div class="dice-roll-block">' in match:
        continue

      match = strip_tags(match)
      for text_skill in df_checks['text_skill']:

        if re.findall(text_skill + " (?:check|roll)", match, re.IGNORECASE):
          dict_skill = {}
          dict_skill.update({'skill': get_5e_skill_name(text_skill.strip()), 'backward_text': match, 'original_name': text_skill.strip()}) 
          skill_list2.append(dict_skill)
          #print('%s - %s' % (text_skill, match))

df_regex_check = pd.DataFrame(skill_list2)

"""
  Step 3: Export the 2 datasets to clean
"""

skills = ['Acrobatics', 'Animal Handling', 'Arcana', 'Athletics', 'Deception', 'History',
          'Insight', 'Intimidation', 'Investigation', 'Medicine', 'Nature', 'Perception',
          'Performance', 'Persuasion', 'Religion', 'Sleight of Hand', 'Stealth', 'Survival'
         ]

# Filters only dnd 5e valid skills
df_message_roll_5e = df_message_roll[df_message_roll['skill'].isin(skills)].copy()
df_regex_check_5e = df_regex_check[df_regex_check['skill'].isin(skills)].copy()

df_message_roll_5e.to_csv ('message_roll_dataset.csv', index = False, header=True)
df_regex_check_5e.to_csv ('regex_check_dataset.csv', index = False, header=True)

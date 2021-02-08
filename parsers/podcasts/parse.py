import re
from pathlib import Path
import os
import pandas as pd
import sys
sys.path.append("..\\..\\utils")
from skills_5e import skills

"""
Returns the regex pattern to find the checks
Checks example: 
  Roll me a deception 
  Give me an acrobatics roll
  Everybody give me a perception check real quick 
"""
def pattern(skill):
  return re.compile("roll(.*?)(?:"+ skill +")|" + skill + " (?:check|roll)", re.IGNORECASE)


# Current Working Directory
cwd = Path(os.getcwd())
data_path = Path(cwd.joinpath('scraped_data/'))

players_char_names = ['Kyle', 'Ali', 'Spurrier', 'Goodrich', 'Yashee', 'Randy', "Raz’ul"]


skill_train_text = []

for text_file in data_path.iterdir():

  df = pd.read_csv(text_file)
  
  for session_transcript in df['selection1_transcript']:

    messages = session_transcript.split(':')

    for message_text in messages:

      for skill in skills():

        for match in re.finditer(pattern(skill), message_text):
          print(match)
    #esse break eh apenas para teste pra manter no primeiro arquivo
    break

#text_skill = 'shirt.Kyle: Roll me a deception.Goodrich: Hello'
#print(text_skill.split(':'))
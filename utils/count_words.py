import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_parquet('../data/rpgai_text.parquet')

skills = ['Deception', 'Intimidation', 'Performance', 'Persuasion', 'Acrobatics', 'Sleight of Hand', 'Stealth', 'Arcana', 'History',
        'Investigation', 'Nature', 'Religion', 'Athletics', 'Animal Handling', 'Insight', 'Medicine', 'Perception', 'Survival'
    ]
    
words_to_df = []

for skill in skills:
    count_vect = CountVectorizer()
    bow = count_vect.fit_transform(df[df['skill'] == skill]['stemm_text'])
    
    count_words = bow.toarray().sum(axis=0)

    for i, word in enumerate(count_vect.get_feature_names()):
        dict_skill = {}
        dict_skill.update({'skill': skill, 'word': word, 'count': count_words[i]}) 
        words_to_df.append(dict_skill)

df_to_export = pd.DataFrame(words_to_df)
df_to_export.to_csv ('word_count.csv', index = False, header=True)
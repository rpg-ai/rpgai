# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:23:50 2021

@author: Thiago Russo
"""
# for NLP preprocessing
import spacy
from unidecode import unidecode
import re
from flashtext import KeywordProcessor

# To get processing time
import time

class NLP_Text_Preprocessor:
   
    # Class Initialization
    def __init__(self):
        
         # Initialize NPL and stemmer
        self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'morphologizer', 'tok2vec', 'attribute_ruler'])
        # Used to check Spacy pipes that are needed and loaded
        #print (self.nlp.pipe_names)
        
        ### Custom Stopwords
        self.rpgai_stopwords = [
                        #Common words
                        "going", "right", "okay", "yeah", "want", "try", "gonna", "good", "yes", "look", "know", "way", "guy",
                        "little", "check", "thin", "thing", "guys", "come", "roll", "let", "time", "got", "maybe", "think", 
                        "fuck", "lot", "shit", "bit", "point",
                        #PCs and NPCs Names
                        "jester", "caleb", "nott", "fjord", "yasha", "beau", "matt", "sam", "travis", "marisha", "ashley", "laura", 
                        "liam", "professor", "thaddeus", "taliesin", "mollymauk", "grog", "pike"
                        ]
        
        ### Create Synonyns dictionary
        
        # Fast replace for multiples strings: stopwords and custom dictionary
        self.keyword_processor = KeywordProcessor()
        
        # Add Spacy Stopwords to processor
        for word in self.nlp.Defaults.stop_words:
            self.keyword_processor.add_keyword(word, ' ')
            
        # Add Custom Stopwords to processor
        for word in self.rpgai_stopwords:
            self.keyword_processor.add_keyword(word, ' ')
       
        pass

    # Method to lower case
    def lower_text(self, corpus):
        # Bring it to lower case
        return [text.lower() for text in corpus]
    
    # Method to replace accents
    def remove_accent(self, corpus):
        # Bring it to lower case
        return [unidecode(text) for text in corpus]
    
    # Method to Keep only alphanumerics
    def clean_text(self, corpus):
        return [re.sub(r'[^a-z0-9]', ' ', text) for text in corpus]
    
    # remove excess spaces
    def strip_extra_space(self, corpus):
        return [re.sub(' +',' ',text).strip() for text in corpus]
    
    # Method to replace stopwords and apply the dictionary for each text in the corpus
    def apply_stopwords_dictionary(self, corpus):
        return [self.keyword_processor.replace_keywords(text) for text in corpus]
    
    # Method to lemmatize the corpus
    ### Struct Colocar suas melhorias aqui!!!
    ### Pode ser necessário ativar os módulos do spacy que estão desativados por conta de performance
    def lemmatizer(self, corpus):
      
        return [' '.join([tok.lemma_ for tok in doc]) for doc in self.nlp.pipe(corpus, batch_size=10000, n_process=1, disable=['ner', 'morphologizer', 'tok2vec', 'attribute_ruler'])]
        #return [' '.join([tok.lemma_ for tok in doc if (tok.pos_ == 'VERB' or tok.pos_ == 'NOUN' or tok.pos_ == 'ADJ')]) 
        #        for doc in self.nlp.pipe(corpus, batch_size=10000, n_process=1, disable=['ner', 'tok2vec'])]

    
    # Make data prep for modeling, pre process and NLP
    def preprocess(self, corpus):
        
        start = time.time()
        corpus = self.lower_text(corpus)
        #print(corpus)
        corpus = self.remove_accent(corpus)
        #print(corpus)
        ### Testar qualidade do modelo sem isso
        corpus = self.clean_text(corpus)
        #print(corpus)
        corpus = self.apply_stopwords_dictionary(corpus)
        #print(corpus)
        corpus = self.strip_extra_space(corpus)
        #print(corpus)
        # corpus = self.lemmatizer(corpus)
        #print(corpus)
        end = time.time()
        print(f'NLP Pre Process Time: {end - start} seconds')
        
        return corpus 


"""
textos = ['I try to identify the holy symbol', 'I try to pick his pockets']

pp = NLP_Text_Preprocessor()
saida = pp.preprocess(textos)

print(saida)

"""
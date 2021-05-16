import re
import csv
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from gensim.models import Phrases
from gensim.corpora import Dictionary, dictionary
import pandas as pd
import DataTypes
import os


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

print('\nImporting csv file...') # Issue: x200b enters the corpus in this configuration.

data_all =[]
data_pre_covid = []
data_post_covid = []

with open('submissionsWithComments.csv', encoding='utf-8') as file:
    csv_reader = csv.reader(file, delimiter = '|')
    for row in csv_reader:
        try:
            dateCreatedUtc = int(row[5])
            s = " ".join([row[1],row[2],row[6]])
            data_all.append(s)
            if dateCreatedUtc < 1583881200:
                data_pre_covid.append(s)
            if dateCreatedUtc > 1583881199:
                data_post_covid.append(s)            
        except:
            pass

print('data has been successfully imported:\n')
print(f'total data:{len(data_all)}\npre-covid:{len(data_pre_covid)}\npost-covid:{len(data_post_covid)}')

def process_data(data):
    
    print('data cleaning and tokenization')
    for idx in range(len(data)):        
        #Remove URLs
        data[idx] = re.sub('http[s]?://\S+', '',data[idx])        
        # Remove new line characters
        data[idx] = re.sub('\s+', ' ',data[idx])        
        # Remove distracting single quotes
        data[idx] = re.sub("\'", "",data[idx])    
        #Tokenize
        data[idx] = data[idx].lower()
        data[idx] = RegexpTokenizer(r'\w+').tokenize(data[idx])
    
    #Lemmatize
    print('Lemmatizing...')
    data = [[WordNetLemmatizer().lemmatize(token) for token in doc] for doc in data]
    
    # Decision: POS-filtering, yes or no?
    #print('Filtering for nouns and verbs...')
    #docs = [[token for (token, pos) in nltk.pos_tag(doc) if pos[:2] == 'NN'
    #or pos[:2] == 'VB'] for doc in docs]

    #Removing irrelevant words
    print('Removing irrelevant words...')
    custom_stopwords = ['amp', 'get', 'use', 'make','would','x200b']
    stpwrds = stopwords.words('english')
    stpwrds.extend(custom_stopwords)
    data = [[token for token in doc if token not in set(stpwrds)
    #and token not in custom_stopwords
    and token.isalnum()
    and not token.isdigit()
    and len(token) > 2] for doc in data]

    #Removing empty documents
    print('Removing empty documents...')
    data = [doc for doc in data if len(doc) > 1]

    return data

def create_corpus_and_dictionary(data, data_type):
    print('Creating dictionary...')    
    dictionary = Dictionary(data)
    
    print('Filtering extremes from dictionary...')
    dictionary.filter_extremes(no_below=20, no_above=0.5)# Decision: which values to use for filtering?
    
    print('Creating bag of words representation...')
    corpus = [dictionary.doc2bow(doc) for doc in data]
    
    print('Saving dictionary and corpus files...')
    dictionary.save(F'raw/dictionary_{data_type}')
    corpora.MmCorpus.serialize(f'raw/corpus_{data_type}', corpus)

print('processing data of the whole period\n')
data_all_processed = process_data(data_all)

print('Processing Pre-covid data....\n')    
data_pre_covid_processed = process_data(data_pre_covid)

print('Processing Post-covid data....\n')    
data_post_covid_processed = process_data(data_post_covid)

if not os.path.exists('raw'):
    os.mkdir('raw')
    
print('creating dictionary and corpus for data')
create_corpus_and_dictionary(data_all_processed, DataTypes.ALL_DATA)
create_corpus_and_dictionary(data_pre_covid_processed,DataTypes.PRE_COVID_DATA)
create_corpus_and_dictionary(data_post_covid_processed,DataTypes.POST_COVID_DATA)

#Saving processed data lists to files
print('Saving processed data lists to files')
with open(f'raw/data_ready_{DataTypes.ALL_DATA}.txt',encoding='utf-8',mode='w') as file:
    file.writelines("%s\n" % topic for topic in data_all_processed)

with open(f'raw/data_ready_{DataTypes.PRE_COVID_DATA}.txt',encoding='utf-8',mode='w') as file:
    file.writelines("%s\n" % topic for topic in data_pre_covid_processed)

with open(f'raw/data_ready_{DataTypes.POST_COVID_DATA}.txt',encoding='utf-8',mode='w') as file:
    file.writelines("%s\n" % topic for topic in data_post_covid_processed)
import csv
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from gensim.models import Phrases
from gensim.corpora import Dictionary

print('\nImporting csv file...') # Issue: x200b enters the corpus in this configuration.
docs = []
with open('submissionsWithComments.csv', encoding='utf-8') as file:
    csv_reader = csv.reader(file, delimiter = '|')
    for row in csv_reader:
        try:
            dateCreatedUtc = int(row[5])
            #if(dateCreatedUtc < 1577833200) :  #Posts before 2020-01-01
            if(1 == 1) :        #All posts                          
            #if(dateCreatedUtc > 1577833199): #posts after 2020-01-01       
                try:
                    docs.append(row[1])
                    docs.append(row[2])
                    docs.append(row[6])
                except IndexError:
                    pass
        except:
            pass
            
print('Removing URLs...')
temp = []
for doc in docs:
    doc = re.sub('http[s]?://\S+', '',  doc)
    temp.append(doc)
docs = temp

print('Tokenizing...')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()
    docs[idx] = RegexpTokenizer(r'\w+').tokenize(docs[idx])

print('Removing all docs with less than 4 words...')
docs = [doc for doc in docs if len(doc) > 4] # Decision: less or more?

print('Lemmatizing...')
docs = [[WordNetLemmatizer().lemmatize(token) for token in doc] for doc in docs]

# Decision: POS-filtering, yes or no?
#print('Filtering for nouns and verbs...')
#docs = [[token for (token, pos) in nltk.pos_tag(doc) if pos[:2] == 'NN'
#or pos[:2] == 'VB'] for doc in docs]

print('Removing irrelevant words...')
#custom_stopwords = ['remote', 'work', 'working', 'would', 'also'] # most pop irrelevant words.
docs = [[token for token in doc if token not in set(stopwords.words('english'))
#and token not in custom_stopwords
and token.isalnum()
and len(token) > 2] for doc in docs]

print('Removing empty documents...')
docs = [doc for doc in docs if len(doc) > 1]

print('Creating dictionary...')
dictionary = Dictionary(docs)

print('Filtering extremes from dictionary...')
dictionary.filter_extremes(no_below=20, no_above=0.5)# Decision: which values to use for filtering?

print('Creating bag of words representation...')
corpus = [dictionary.doc2bow(doc) for doc in docs]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

print('Saving dictionary and corpus to file...')
dictionary.save('dictionary')
corpora.MmCorpus.serialize('corpus', corpus)

print('Preprocessing complete. To perform topic modeling, run LDAmodeling.py.\n')

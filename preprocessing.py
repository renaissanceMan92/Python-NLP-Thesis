import csv
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from gensim.models import Phrases
from gensim.corpora import Dictionary

print('\nImporting csv file...')
docs = []
with open('posts.csv', encoding='ISO 8859-1') as file:
    csv_reader = csv.reader(file, delimiter = '|')
    for row in csv_reader:
        docs.append(row[0])
        try:
            docs.append(row[1])
        except IndexError:
            pass

print('Removing URLs...')
temp = []
for doc in docs:
    doc = re.sub('http[s]?://\S+', '',  doc)
    temp.append(doc)
docs = temp

print('Tokenizing...')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = RegexpTokenizer(r'\w+').tokenize(docs[idx])  # Split into words.

print('Removing all docs with less than 3 words...')
docs = [doc for doc in docs if len(doc) > 4]

print('Lemmatizing...')
docs = [[WordNetLemmatizer().lemmatize(token) for token in doc] for doc in docs]

print('Filtering for nouns and verbs...')
docs = [[token for (token, pos) in nltk.pos_tag(doc) if pos[:2] == 'NN'
or pos[:2] == 'VB'] for doc in docs]

print('Removing stop words etc...')
docs = [[token for token in doc if token not in set(stopwords.words('english'))
and token.isalnum()
and len(token) > 3] for doc in docs]

print('Removing empty documents...')
docs = [doc for doc in docs if len(doc) > 1]

print('Creating dictionary...')
dictionary = Dictionary(docs)

print('Filtering extremes from dictionary...')
dictionary.filter_extremes(no_below=20, no_above=0.5)

print('Creating bag of words representation...')
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Alternative TF-IDF:
#from gensim import corpora, models
#tfidf = models.TfidfModel(bow_corpus)
#corpus_tfidf = tfidf[bow_corpus]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

print('Saving dictionary and corpus to file...')
dictionary.save('dictionary')
corpora.MmCorpus.serialize('corpus', corpus)

print('Preprocessing complete. To perform topic modeling, run LDAmodeling.py.\n')
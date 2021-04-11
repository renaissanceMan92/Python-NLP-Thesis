
import csv
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# import original corpus
docs = []
with open('posts.csv', encoding='ISO 8859-1') as file:
    csv_reader = csv.reader(file, delimiter = '|')
    for row in csv_reader:
        docs.append(row[0])
        try:
            docs.append(row[1])
        except IndexError:
            pass

# remove links.
temp = []
for doc in docs:
    doc = re.sub('http[s]?://\S+', '',  doc)
    temp.append(doc)
docs = temp

# Tokenize the documents.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

# Remove all non-nouns from the corpus.
docs = [[token for (token, pos) in nltk.pos_tag(doc) if(pos[:2] == 'NN')] for doc in docs]

# remove stop words, non-alphabetic words and short words.
en_stops = set(stopwords.words('english'))
custom_stops = ['amp']
docs = [[token for token in doc if token not in en_stops
and token not in custom_stops
and re.match("^[A-Za-z_-]*$", token)
and len(token) > 1] for doc in docs]

# remove all documents with 5 words or less.
#docs = [doc for doc in docs if len(doc) > 1]

# Lemmatize the documents.
lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

# Compute bigrams.
bigram = Phrases(docs, min_count=5)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            docs[idx].append(token)

# Create a dictionary
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)

# Create a bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]

# print the number of unique words and the number of documents.
print('\nNumber of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

temp = dictionary[0]  # This is only to "load" the dictionary.

num_topics = 10
    # run LDA model.
lda_model = LdaModel(
    corpus = corpus,
    id2word = dictionary.id2token,
    chunksize = 4000,
    alpha = 'auto',
    eta = 'auto',
    iterations = 50,
    num_topics = num_topics,
    passes = 20,
    eval_every = None
)
topics = lda_model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
pprint(topics)
avg_topic_coherence = sum([t[1] for t in topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

# Create interactive HTML-doc for visualizing topics.
visualisation = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')
print("visualisation created.")



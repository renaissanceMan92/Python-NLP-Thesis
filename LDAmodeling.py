from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import corpora
from pprint import pprint
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

print('\nImporting corpus and dictionary...')
dictionary = corpora.Dictionary.load('dictionary')
corpus = corpora.MmCorpus('corpus')

# Decision: use TF-IDF? How, why
#tfidf = models.TfidfModel(corpus)
#corpus_tfidf = tfidf[corpus]

print('Performing topic modeling...')
temp = dictionary[0]
num_topics = 20 # Decision: how many topics?
lda_model = LdaModel(
    corpus = corpus,
    id2word = dictionary.id2token,
    chunksize = 4000, # decision: what chunksize?
    alpha = 'auto',
    eta = 'auto',
    iterations = 50, # decision: how many iterations?
    num_topics = num_topics,
    passes = 20, # decision: how many passes?
    eval_every = None,
)
topics = lda_model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
print('Calculating coherence...')
avg_topic_coherence = sum([t[1] for t in topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

print('Creating HTML visualization...')
visualisation = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

print("The program is completed.\n")

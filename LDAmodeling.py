
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import corpora
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

print('\nImporting corpus and dictionary...')
dictionary = corpora.Dictionary.load('dictionary')
corpus = corpora.MmCorpus('corpus')

print('Performing topic modeling...')
temp = dictionary[0]  # This is only to "load" the dictionary.
num_topics = 30
    # run LDA model.
lda_model = LdaModel(
    corpus = corpus, # alternatively change to corpus_tfidf.
    id2word = dictionary.id2token,
    chunksize = 4000,
    alpha = 'auto',
    eta = 'auto',
    iterations = 100,
    num_topics = num_topics,
    passes = 20,
    eval_every = None
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

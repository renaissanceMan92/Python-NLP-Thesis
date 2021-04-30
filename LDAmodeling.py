from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import corpora
from pprint import pprint
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation,strip_numeric
import os

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

#Create folder for all output files
os.makedirs('Output/Models',exist_ok=True)

#save the model
lda_model.save('Output/Models/trained_model')

#Get top topics
topics = lda_model.top_topics(corpus)

# Save topics to file topics_raw.txt
topics_file_raw = open('Output/topics_raw.txt','w')
for idx, topic in lda_model.print_topics():
    s = "Topic: {} \nWords: {}".format(idx,topic)
    topics_file_raw.write(s)
    topics_file_raw.write('\n')
topics_file_raw.close()

#Save topics formatted
topics_formatted = open('Output/topics_formatted.txt','w')
for idx,topic in lda_model.show_topics(formatted=False,num_topics=20):
    topics_formatted.write('Topic: {} \nWords: {}\n'.format(idx, [w[0] for w in topic]))
topics_formatted.close()

#Save top topics
top_topics = open('Output/top_topics.txt','w')
top_topics.write('Top topics:\n')
for topic in topics:
    words =[]
    for k,v in topic[0]:
        words.append(v)
    joined = ", ".join(words)
    top_topics.write("Topic {}:\n[{}]\n".format(topics.index(topic),joined))
    
top_topics.close()


# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
print('Calculating coherence...')
avg_topic_coherence = sum([t[1] for t in topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

print('Creating HTML visualization...')
visualisation = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(visualisation, 'Output/LDA_Visualization.html')
pyLDAvis.save_json(visualisation,'Output/LDA_Visualization.json')


print('Saving logfile...')
log_file = open('Output/logfile.txt', 'w')
log_file.write('\nNumber of unique tokens in dictionary (after filtering): %d' % len(dictionary))
log_file.write('\nNumber of documents in corpus: %d' % len(corpus))
log_file.write('\nNumber of topics: %d' % num_topics)
log_file.write('\nNumber of iterations: %d' % lda_model.iterations)
log_file.write('\nAverage topic coherence: %.4f.' % avg_topic_coherence)
log_file.close()

print("The program is completed.\n")

from collections import Counter
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import corpora
from pprint import pprint
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.parsing.preprocessing import STOPWORDS, preprocess_string, strip_punctuation,strip_numeric
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import matplotlib.colors as mcolors
import DataTypes
from gensim.models import CoherenceModel


data_type=DataTypes.POST_COVID_DATA

data_ready= []
with open(f'raw/data_ready_{data_type}.txt','r',encoding='UTF-8') as file:
    for line in file.readlines():
        current = line[:-1]
        data_ready.append(current)


dictionary = corpora.Dictionary.load(f'raw/dictionary_{data_type}')
corpus = corpora.MmCorpus(f'raw/corpus_{data_type}')

# lda_model = LdaModel(
#         corpus = corpus,
#         id2word = dictionary.id2token,
#         chunksize = 4000, # decision: what chunksize?
#         alpha = 'auto',
#         eta = 'auto',
#         iterations = 50, # decision: how many iterations?
#         num_topics = 40,
#         passes = 20, # decision: how many passes?
#         eval_every = None,
#     )

# if __name__ == "__main__":
#     coherence_model_lda = CoherenceModel(model=lda_model, texts=data_ready,corpus=corpus,  coherence='u_mass')
#     coherence_lda = coherence_model_lda.get_coherence()
#     print('\nCoherence Score: ', coherence_lda)

####################################
def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    values= dict()
    for num_topics in range(start, limit, step):
        print(f'testing with {num_topics} topics\n')
        temp = dictionary[0] # This is only to "load" the dictionary.
        model = LdaModel(
        corpus = corpus,
        id2word = dictionary.id2token,
        chunksize = 4000, # decision: what chunksize?
        alpha = 'auto',
        eta = 'auto',
        iterations = 1000, # decision: how many iterations?
        num_topics = num_topics,
        passes = 100, # decision: how many passes?
        eval_every = None,
        )
        avg_topic_coherence = sum([t[1] for t in model.top_topics(corpus)]) / num_topics
        values[num_topics] = avg_topic_coherence
        
        # print('now trying coherence with c_v')
        # # if __name__ == "__main__":
        # coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        # coherence_model =coherencemodel.get_coherence()
        # print(coherence_model)
        

    return values

values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=data_ready, start=10, limit=41, step=5)
for k,v in values.items():
    print(f'{k} topics has coherence {v}')
plt.plot(values.keys(),values.values())
plt.xlabel('Nr. of topics')
plt.ylabel('Coherence score')
plt.legend(("values.values"), loc='best')
plt.show()

from gensim.models import LdaModel
from gensim import corpora
import DataTypes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



data_type = DataTypes.POST_COVID_DATA
lda_model = LdaModel.load(f'Output/{data_type}/Models/trained_model')
dictionary = corpora.Dictionary.load(f'raw/dictionary_{data_type}')
corpus = corpora.MmCorpus(f'raw/corpus_{data_type}')
n_words = 10

topic_words = pd.DataFrame({})

for i, topic in enumerate(lda_model.get_topics()):
    top_feature_ids = topic.argsort()[-n_words:][::-1]
    feature_values = topic[top_feature_ids]
    words = [dictionary[id] for id in top_feature_ids]
    topic_df = pd.DataFrame({'value': feature_values, 'word': words, 'topic': i})
    topic_words = pd.concat([topic_words, topic_df], ignore_index=True)


g = sns.FacetGrid(topic_words, col="topic", col_wrap=3, sharey=False)
g.map(plt.barh, "word", "value")
plt.show()



from gensim.models import LdaModel
import gensim.corpora as corpora
import DataTypes
import pandas as pd

'''
Function to create Coherene_per_topic tabel
'''
data_type = DataTypes.PRE_COVID_DATA
lda_model = LdaModel.load(f'Output/{data_type}/Models/trained_model')

dictionary = corpora.Dictionary.load(f'raw/dictionary_{data_type}')
corpus = corpora.MmCorpus(f'raw/corpus_{data_type}')

rows=[]
topics = lda_model.top_topics(corpus)
for topic in topics:
    words = []
    for k,v in topic[0]:
        words.append(v)
    joined = ", ".join(words)
    rows.append([topics.index(topic),topic[1],words])
sent_items_df = pd.DataFrame(rows, columns=['Topic No.','Topic Coherence','Topic words'])


text_file = open(f"Output/{data_type}/coherence_per_topic.html", "w",encoding='UTF-8')
text_file.write(sent_items_df.to_html())
text_file.close()



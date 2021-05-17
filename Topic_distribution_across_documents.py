from gensim.models import LdaModel
from gensim import corpora
import pandas as pd
import DataTypes

data_type = DataTypes.PRE_COVID_DATA

lda_model = LdaModel.load(f'Output/{data_type}/Models/trained_model')

dictionary = corpora.Dictionary.load(f'raw/dictionary_{data_type}')
corpus = corpora.MmCorpus(f'raw/corpus_{data_type}')

#Import list of tokens (word lists) for this data type.
data_ready= []
with open(f'raw/data_ready_{data_type}.txt','r',encoding='UTF-8') as file:
    for line in file.readlines():
        current = line[:-1]
        data_ready.append(current)

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=None):
    #Prepares and format the topics in a data frame.
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
text_file = open(f"Output/{data_type}/Topic_distribution_across_documents.html", "w", encoding='UTF-8')
text_file.write(df_dominant_topics.to_html())
text_file.close()
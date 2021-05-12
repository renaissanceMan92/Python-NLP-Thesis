from wordcloud import WordCloud, STOPWORDS
from gensim.models import LdaModel
import DataTypes
import ast
from collections import Counter
data_type = DataTypes.PRE_COVID_DATA
lda_model = LdaModel.load(f'Output/{data_type}/Models/trained_model')

# topics_dict={}
# for i,t in lda_model.show_topics(formatted = False):
#     for word,weight in t:
#         topics_dict[word]=weight 
# wc_all_topics = WordCloud(max_words= 50,stopwords =STOPWORDS,width=1600,height=800)
# wc_all_topics.fit_words(dict(topics_dict))
# wc_all_topics.to_file(f'Output/{data_type}/WordClouds/wordcloud_all_topics.png')


######
#word cloud by frequency

#load data
data_ready= []
with open(f'raw/data_ready_{data_type}.txt','r',encoding='UTF-8') as file:
    for line in file.readlines():
        current = line[:-1]
        data_ready.append(current)
data_as_words_list = []
#Convert topic from string to list of words
for topic in data_ready:
    topic_to_words =ast.literal_eval(topic)
    for word in topic_to_words:
        data_as_words_list.append(word) #Add each word to list of words
        
counts= Counter(data_as_words_list) #Count frequency of each word in list
wc_all_topics_by_frequency = WordCloud(max_words=50,stopwords=STOPWORDS,width=1600,height=800)    
wc_all_topics_by_frequency.fit_words(dict(counts.most_common(100))) #Take 100 most common words, take best 50 after filtering with stopwords
print(counts.most_common(100))
wc_all_topics_by_frequency.to_file(f'Output/{data_type}/WordClouds/wordcloud_all_topics_by_frequency.png')
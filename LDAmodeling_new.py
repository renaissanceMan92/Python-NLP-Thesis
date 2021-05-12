from collections import Counter
from gensim.corpora import Dictionary
from gensim.models import LdaModel, coherencemodel
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
from multiprocessing import Process, freeze_support


def run_modelling(data_type):
    #This functions run all the methods to perform topic modelling and save its outputs.
    #Parameter: data_type is a string of data type as in DataTypes.py (post covid or pre covid)
    
    #Create folder for all output files
    os.makedirs(f'Output/{data_type}/Models',exist_ok=True)
    if not os.path.exists(f'Output/{data_type}/WordClouds'):
        os.mkdir(f'Output/{data_type}/WordClouds')
        
    #Import dictionary and corpus for this data.
    print('\nImporting corpus and dictionary...')
    dictionary = corpora.Dictionary.load(f'raw/dictionary_{data_type}')
    corpus = corpora.MmCorpus(f'raw/corpus_{data_type}')

    #Import list of tokens (word lists) for this data type.
    data_ready= []
    with open(f'raw/data_ready_{data_type}.txt','r',encoding='UTF-8') as file:
        for line in file.readlines():
            current = line[:-1]
            data_ready.append(current)
            
    #Decision: use TF-IDF? How, why
    #tfidf = models.TfidfModel(corpus)
    #corpus_tfidf = tfidf[corpus]

    #Building the model.
    #Creates the model with prespecified number of topics, iterations and passes.
    print('Performing topic modeling...')
    temp = dictionary[0] # This is only to "load" the dictionary.
    num_topics = 35 # Decision: how many topics?
    lda_model = LdaModel(
        corpus = corpus,
        id2word = dictionary.id2token,
        chunksize = 4000, # decision: what chunksize?
        alpha = 'auto',
        eta = 'auto',
        iterations = 100, # decision: how many iterations?
        num_topics = num_topics,
        passes = 100, # decision: how many passes?
        eval_every = None,
    )
    #Get top topics
    topics = lda_model.top_topics(corpus)

    ##########################################################
    #This section is to get the dominant topic per each sentence.
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

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show

    text_file = open(f"Output/{data_type}/dominant_topic_per_sentence.html", "w", encoding='UTF-8')
    text_file.write(df_dominant_topic.to_html())
    text_file.close()
    # print(df_dominant_topic.head(10))

    ##########################################################
    #This section is to get the most representative documents for each topic
    # Group top 5 sentences under each topic
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                                axis=0)

    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

    # Show
    text_file = open(f"Output/{data_type}/most_representative_document.html", "w",encoding='UTF-8')
    text_file.write(sent_topics_sorteddf_mallet.to_html())
    text_file.close()

    ###############################################################
       
    '''
    Section: Save output and other files.
    '''
    # save the model
    lda_model.save(f'Output/{data_type}/Models/trained_model')

    #Uses a topics_dict to create a word cloud of all topics within
    topics_dict={}
    for i,t in lda_model.show_topics(formatted = False):
        for word,weight in t:
            topics_dict[word]=weight 
    wc_all_topics = WordCloud(max_words= 50,stopwords =STOPWORDS, width=1600,height=800)
    wc_all_topics.fit_words(dict(topics_dict))
    wc_all_topics.to_file(f'Output/{data_type}/WordClouds/wordcloud_all_topics.png')
    #to show it:
    # plt.imshow(wc_all_topics, interpolation ='bilinear')
    # plt.axis('off')
    # plt.show()

    #Create word cloud for each topic
    for t in range(lda_model.num_topics):
        wc = WordCloud(background_color="white", max_words = 20, stopwords=STOPWORDS)
        wc.fit_words(dict(lda_model.show_topic(t, 200)))
        wc.to_file(f'Output/{data_type}/WordClouds/wordcloud_topic_{str(t)}.png')
        
        #To Show word clouds with matplot.pyplot
        # plt.figure()
        # plt.imshow(wc, interpolation ='bilinear')
        # plt.axis('off')
        # plt.title("Topic #" + str(t))
        # plt.show()

    # Save topics to file topics_raw.txt
    topics_file_raw = open(f'Output/{data_type}/topics_raw.txt','w')
    for idx, topic in lda_model.print_topics():
        s = "Topic: {} \nWords: {}".format(idx,topic)
        topics_file_raw.write(s)
        topics_file_raw.write('\n')
    topics_file_raw.close()

    #Save topics formatted
    topics_formatted = open(f'Output/{data_type}/topics_formatted.txt','w')
    for idx,topic in lda_model.show_topics(formatted=False,num_topics=20):
        topics_formatted.write('Topic: {} \nWords: {}\n'.format(idx, [w[0] for w in topic]))
    topics_formatted.close()

    #Save top topics
    top_topics = open(f'Output/{data_type}/top_topics.txt','w',encoding='UTF-8')
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
    pyLDAvis.save_html(visualisation, f'Output/{data_type}/LDA_Visualization.html')
    pyLDAvis.save_json(visualisation,f'Output/{data_type}/LDA_Visualization.json')

    print('Saving logfile...')
    log_file = open(f'Output/{data_type}/logfile.txt', 'w')
    log_file.write(f'Log for {data_type}:')
    log_file.write('\nNumber of unique tokens in dictionary (after filtering): %d' % len(dictionary))
    log_file.write('\nNumber of documents in corpus: %d' % len(corpus))
    log_file.write('\nNumber of topics: %d' % num_topics)
    log_file.write('\nNumber of iterations: %d' % lda_model.iterations)
    log_file.write('\nAverage topic coherence: %.4f.' % avg_topic_coherence)
    log_file.write(f'\nModel perplexity: {lda_model.log_perplexity(corpus)}')
    log_file.close()

    # print('============================================')
    # print(f"The program is completed for {data_type}.")
    # print('============================================')


#ٌRun topic modelling 10 times to compare and get best model. 
for x in range(0,10):
    print('============================')
    print(f'Running iteration nr.{x}')
    print('============================')
    run_modelling(DataTypes.POST_COVID_DATA)
    run_modelling(DataTypes.PRE_COVID_DATA)
    os.rename('Output', f'Output_{x}')
    print(f'done with iteration nr.{x}')

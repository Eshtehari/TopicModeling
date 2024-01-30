import multiprocessing
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from settings import Settings
from tqdm import tqdm

sys.path.append("..")
from topic_modeling_diffusion.src.topicmodeling.preprocess import TextProcessor

csv_file = Settings.csv_file
body_column = Settings.body_column
category_column = Settings.category_column
title_column = Settings.title_column
predicted_csv_file = Settings.predicted_csv_file
lda_file = Settings.lda_file
dictionary_file = Settings.dictionary_file

lda = LdaModel.load(lda_file)

news = pd.read_csv(csv_file, dtype=str)
# news_train = news[0 : int(len(news) * 0.8)]
# news_test = news[int(len(news) * 0.8) : len(news)]
# print("Len of news:", len(news))
# print("Len of news_train:", len(news_train))
# print("Len of news_test:", len(news_test))

dictionary = Dictionary.load(dictionary_file)
text_processor = TextProcessor([""])

# print(lda.show_topics())
# print(len(lda.show_topics()))

news = news.loc[0:100]

def single_process():
    lemmatized_sentences = []
    corpus_list = []
    predictions = []

    for index, row in tqdm(news.iterrows(), total=len(news)):
        description = row[body_column]

        # Pre-process
        lemmatized_sentence = text_processor.worker([description])
        lemmatized_sentences.append(lemmatized_sentence)

        corpus = dictionary.doc2bow(lemmatized_sentence[0])
        corpus_list.append(corpus)
        vector = lda[corpus]

        # Same method: lda.get_document_topics is vector
        # print(lda.get_document_topics(bow=corpus))
        # print(vector)

        predicted_category = max(vector, key=lambda x: x[1])[0]
        # print(predicted_category)
        # print(row["category"])
        # news.loc[index, "prediction"] = predicted_category
        predictions.append(predicted_category)

    return lemmatized_sentences, corpus_list, predictions


def worker(list_of_indexes):
    lemmatized_sentences = []
    corpus_list = []
    predictions = []

    for index in list_of_indexes:
        description = news.loc[index, body_column]

        # Pre-process
        lemmatized_sentence = text_processor.worker([description])
        lemmatized_sentences.append(lemmatized_sentence[0])

        corpus = dictionary.doc2bow(lemmatized_sentence[0])
        corpus_list.append(corpus)
        vector = lda[corpus]

        predicted_category = max(vector, key=lambda x: x[1])[0]
        predictions.append(predicted_category)

    return lemmatized_sentences, corpus_list, predictions


def multi_process():
    num_processes = multiprocessing.cpu_count()
    pool = Pool(num_processes)
    data_chunks = np.array_split(range(0, len(news)), num_processes)
    # results = pool.map(worker, data_chunks)
    results = list(tqdm(pool.imap(worker, data_chunks), total=len(news)))
    pool.close()
    pool.join()
    
    # mock = [("lemmatized_sentences", "corpus_list", "predictions"), ("lemmatized_sentences", "corpus_list", "predictions"), ...]
    lemmatized_sentences = [r[0] for r in results]
    lemmatized_sentences = [j for i in lemmatized_sentences for j in i]
    corpus_list = [r[1] for r in results]
    corpus_list = [j for i in corpus_list for j in i]
    predictions = [r[2] for r in results]
    predictions = [j for i in predictions for j in i]
    
    return lemmatized_sentences, corpus_list, predictions

lemmatized_sentences, corpus_list, predictions = multi_process()

top_topics = lda.top_topics(texts=lemmatized_sentences, dictionary=dictionary, coherence="c_v")
top_topics = [topic[0] for topic in top_topics]
print(top_topics)
print(len(top_topics))
# print(len(top_topics[0]))
for topic_index, topic in enumerate(top_topics):
    most_representative_topic = max(topic, key=lambda x: x[0])[1]
    print(topic_index, "\t"+news.loc[topic_index, title_column], "\t["+news.loc[topic_index, category_column]+"]")

news["prediction"] = predictions
news["preprocessed_doc"] = lemmatized_sentences

news.to_csv(predicted_csv_file)
print("Saved predicted csv in", predicted_csv_file)

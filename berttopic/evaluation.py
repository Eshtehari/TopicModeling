import csv
import sys

import numpy as np
import pandas as pd
import sklearn.metrics
from bertopic import BERTopic
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

sys.path.append("..")
from diversity_metrics import TopicDiversity
from lda.settings import Settings
from topic_modeling_diffusion.src.topicmodeling.preprocess import TextProcessor

csv_file = Settings.csv_file
body_column = Settings.body_column
category_column = Settings.category_column
dictionary_file = Settings.dictionary_file

topic_model = BERTopic.load("bertmodel.bin")
news = pd.read_csv(csv_file, dtype=str)

list_of_sentences = []

with open(csv_file, "r", encoding="utf-8") as fd:
    reader = csv.DictReader(fd)
    for row in reader:
        list_of_sentences.append(row[body_column])

# Pre-process
text_processor = TextProcessor(list_of_sentences)
del list_of_sentences
text_processor.lemmatize_sentences()
lemmatized_sentences = text_processor.lemmatized_sentences
lemmatized_sentences_str = [' '.join(x) for x in lemmatized_sentences]


print("Topics:")
print(topic_model.generate_topic_labels())

print("Probs:")
print(topic_model.probabilities_)
print("")

print("Topic labels:")
print(topic_model.topic_labels_)
print("")

print("Topic info:")
print(topic_model.get_topic_info())
print("")

print("Topic terms:")
print(topic_model.get_topics())
print("")

print("Topic 0:")
print(topic_model.get_topic(0))
print("")

print("Document info:")
print(topic_model.get_document_info(lemmatized_sentences_str))
print("")

# print("Visualize topics:")
# print(topic_model.visualize_topics())
# print("")

# print("Visualize docs:")
# print(topic_model.visualize_documents(lemmatized_sentences_str))
# print("")

news['prediction'] = topic_model.topics_
news['prediction'] = news['prediction'].apply(lambda x: int(float(x)))

## from p4_evaluation.py
# Purity Score
news_gb = (
    news.groupby(["prediction", category_column])[body_column].count().reset_index()
)
print(news_gb)
news_max = news_gb.groupby(["prediction"])[body_column].idxmax()
print(news_max)

# Returns the corresponding values of news_gb based on index of news_max
news_res = news_gb.loc[news_max]
print(news_res)

# Grabs index from news_res instead
news_res = pd.merge(
    news_res,
    news_gb[["prediction", category_column]],
    on=["prediction", category_column],
    how="left",
)
print(news_res)

print("Purity: {:.2f}%".format(news_res[body_column].sum() * 100 / len(news)))

# Topic Coherence (TC) Score
dictionary = Dictionary.load(dictionary_file)

## from p2_lda.py

# To bag-of-words
dictionary = Dictionary(lemmatized_sentences)
corpus = [dictionary.doc2bow(text) for text in lemmatized_sentences]
## end of p2_lda.py

# coherence_model_lda = CoherenceModel(
#     model=topic_model, texts=lemmatized_sentences, dictionary=dictionary, coherence="c_v"
# )
topics_terms = list(topic_model.get_topics().values())
topics = [[term for term, prob in topic] for topic in topics_terms]
coherence_model_lda = CoherenceModel(
    topics=topics,
    texts=lemmatized_sentences,
    dictionary=dictionary,
    coherence="c_v",
)
coherence_lda = coherence_model_lda.get_coherence()
print("Topic Coherence Score:", coherence_lda)

# Topic Diversity (TD) Score
# ...

nmi = sklearn.metrics.normalized_mutual_info_score(news[category_column].astype('category').cat.codes, news['prediction'])
print("Normalized Mutual Info Score", nmi)

rni = sklearn.metrics.rand_score(news[category_column].astype('category').cat.codes, news['prediction'])
print("Rand Index Score", rni)

f1 = sklearn.metrics.f1_score(news[category_column].astype('category').cat.codes, news['prediction'], average='macro')
print("F1 (macro) Score", f1)

f1 = sklearn.metrics.f1_score(news[category_column].astype('category').cat.codes, news['prediction'], average='micro')
print("F1 (micro) Score", f1)

f5 = sklearn.metrics.fbeta_score(news[category_column].astype('category').cat.codes, news['prediction'], average='macro', beta=5)
print("F5 (beta=5) (macro) Score", f5)

fowlkes_mallows_score = sklearn.metrics.fowlkes_mallows_score(news[category_column].astype('category').cat.codes, news['prediction'])
print("Fowlkes-Mallows Score", fowlkes_mallows_score)
## end of p4_evaluation.py

# OCTIS
topic_diversity = TopicDiversity(topk=10)
topic_diversity_score = topic_diversity.score(topic_model, model='bertopic')
print("Topic diversity:", topic_diversity_score)

import csv
import sys

import numpy as np
import pandas as pd
import sklearn.metrics
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
# from octis.evaluation_metrics.coherence_metrics import Coherence
# from octis.evaluation_metrics.diversity_metrics import TopicDiversity
# from gensim.models import CoherenceModel
from settings import Settings

sys.path.append("..")
from topic_modeling_diffusion.src.topicmodeling.preprocess import TextProcessor
from diversity_metrics import TopicDiversity

predicted_csv_file = Settings.predicted_csv_file
category_column = Settings.category_column
body_column = Settings.body_column
lda_file = Settings.lda_file
dictionary_file = Settings.dictionary_file
csv_file = Settings.csv_file
body_column = Settings.body_column

news = pd.read_csv(predicted_csv_file, dtype=str)
news['prediction'] = news['prediction'].apply(lambda x: int(float(x)))

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
lda = LdaModel.load(lda_file)
dictionary = Dictionary.load(dictionary_file)

list_of_sentences = []

## from p2_lda.py
with open(csv_file, "r", encoding="utf-8") as fd:
    reader = csv.DictReader(fd)
    for row in reader:
        list_of_sentences.append(row[body_column])

# Pre-process
print("Length of list_of_sentences:", len(list_of_sentences))
print("Size of list_of_sentences:", sys.getsizeof(list_of_sentences))
text_processor = TextProcessor(list_of_sentences)
del list_of_sentences
text_processor.lemmatize_sentences()
lemmatized_sentences = text_processor.lemmatized_sentences

# To bag-of-words
dictionary = Dictionary(lemmatized_sentences)
corpus = [dictionary.doc2bow(text) for text in lemmatized_sentences]
## end of p2_lda.py

# coherence_model_lda = CoherenceModel(
#     model=lda, texts=list_of_sentences, dictionary=dictionary, coherence="c_v"
# ) # nan
coherence_model_lda = CoherenceModel(
    model=lda, texts=lemmatized_sentences, dictionary=dictionary, coherence="c_v"
) # Coherence Score: 0.5304070691426127
# coherence_model_lda = CoherenceModel(
#     model=lda, corpus=corpus, dictionary=dictionary, coherence="u_mass"
# ) # Coherence Score: -4.325622798949165
coherence_lda = coherence_model_lda.get_coherence()
print("Topic Coherence Score:", coherence_lda)

# Topic Diversity (TD) Score
# topics_matrix = lda.get_topics()
# topics_matrix.T

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

# OCTIS
# npmi = Coherence(texts=lemmatized_sentences, topk=10, measure='c_v')
# topic_diversity = TopicDiversity(topk=10)

# print('TC by OCTIS:', npmi)
# print('TD by OCTIS:', topic_diversity)

topic_diversity = TopicDiversity(topk=10)
topic_diversity_score = topic_diversity.score(lda, dictionary, lemmatized_sentences)
print("Topic diversity:", topic_diversity_score)

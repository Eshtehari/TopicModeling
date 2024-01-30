import sys

import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from settings import Settings
from tqdm import tqdm

sys.path.append("..")
from topic_modeling_diffusion.src.topicmodeling.preprocess import TextProcessor

csv_file = Settings.csv_file
body_column = Settings.body_column
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

print(lda.top_topics(corpus_list, dictionary=dictionary, coherence='c_v'))

news["prediction"] = predictions
news["preprocessed_doc"] = lemmatized_sentences

news.to_csv(predicted_csv_file)
print("Saved predicted csv in", predicted_csv_file)

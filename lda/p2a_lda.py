import csv
import sys

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from settings import Settings

sys.path.append("..")
from topic_modeling_diffusion.src.topicmodeling.preprocess import TextProcessor

csv_file = Settings.csv_file
body_column = Settings.body_column
lda_file = Settings.lda_file
dictionary_file = Settings.dictionary_file

list_of_sentences = []

with open(csv_file, "r", encoding="utf-8") as fd:
    reader = csv.DictReader(fd)
    for row in reader:
        list_of_sentences.append(row[body_column])

# Take .8 for train
# list_of_sentences = list_of_sentences[0 : int(len(list_of_sentences) * 0.8)]

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

# Train LDA
lda = LdaModel(corpus, num_topics=14)

lda.save(lda_file)
print("Saved model in", lda_file)

dictionary.save(dictionary_file)
print("Dictionary saved in", dictionary_file)

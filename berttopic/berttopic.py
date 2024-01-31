import csv
import os
import sys

from bertopic import BERTopic

sys.path.append("..")
from lda.settings import Settings
from topic_modeling_diffusion.src.topicmodeling.preprocess import TextProcessor

csv_file = Settings.csv_file
body_column = Settings.body_column

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
lemmatized_sentences = [' '.join(x) for x in lemmatized_sentences]

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

topic_model = BERTopic(nr_topics=14)
topics, probs = topic_model.fit_transform(lemmatized_sentences)

topic_model.save("bertmodel.bin")
print("Saved model in", "bertmodel.bin")

print("Topics:")
print(topics)
print("")

print("Probs:")
print(probs)
print("")

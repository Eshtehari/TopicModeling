import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from settings import Settings

csv_file = Settings.csv_file
lda_file = Settings.lda_file
dictionary_file = Settings.dictionary_file
category_column = Settings.category_column
body_column = Settings.body_column

lda = LdaModel.load(lda_file)
dictionary = Dictionary.load(dictionary_file)

print("Topics:")
print(len(lda.get_topics()))
print(len(lda.get_topics()[0]))
print(lda.get_topics())
print("")

# print("Term topics:")
# print(lda.get_term_topics(0))
# print("")

print("Topic terms:")
for i in range(0, 14):
    topic_terms = lda.get_topic_terms(i)
    topic_terms = [(dictionary[idx], prob) for idx, prob in topic_terms]
    print(topic_terms)
print("")

# print(dictionary.token2id['hello'])

news = pd.read_csv(csv_file, dtype=str)
# print(news[category_column].value_counts())

# Topic 0 words. I ran codes again and the topics changed!
# print(
#     list(
#         set(
#             news[
#                 news[body_column].str.contains("player", case=False)
#                 & news[body_column].str.contains("los angeles", case=False)
#                 & news[body_column].str.contains("film", case=False)
#                 & news[body_column].str.contains("movie", case=False)
#                 & news[body_column].str.contains("hollywood", case=False)
#             ][category_column]
#         )
#     )
# )

# 5:
# [('minister', 0.031022964), ('prime', 0.020043587), ('election', 0.017693048), ('president', 0.014308473), ('party', 0.0133742215), ('afp', 0.012461479), ('say', 0.011409829), ('government', 0.010364545), ('leader', 0.009880761), ('country', 0.009671236)]
print(
    list(
        set(
            news[
                news[body_column].str.contains("prime", case=False)
                & news[body_column].str.contains("minister", case=False)
                & news[body_column].str.contains("election", case=False)
                & news[body_column].str.contains("president", case=False)
                & news[body_column].str.contains("government", case=False)
                & news[body_column].str.contains("leader", case=False)
                & news[body_column].str.contains("country", case=False)
            ][category_column]
        )
    )
)

# 12:
# [('film', 0.013655206), ('year', 0.013241977), ('los', 0.012666811), ('angeles', 0.012355394), ('star', 0.012341741), ('reuters', 0.010129233), ('award', 0.0084549375), ('movie', 0.008342363), ('title', 0.007675141), ('first', 0.0075670853)]
print(
    list(
        set(
            news[
                news[body_column].str.contains("los", case=False)
                & news[body_column].str.contains("angeles", case=False)
                & news[body_column].str.contains("star", case=False)
                & news[body_column].str.contains("award", case=False)
                & news[body_column].str.contains("movie", case=False)
            ][category_column]
        )
    )
)

import re

import matplotlib.pyplot as plt
import pandas as pd
from settings import Settings

orig_csv_file = Settings.orig_csv_file
csv_file = Settings.csv_file
body_column = Settings.body_column
category_column = Settings.category_column

with open(orig_csv_file, "r", encoding="latin-1") as fd:
    text = fd.read()

rows = text.split("\\N\n")
rows = [row.replace("\\\n", "") for row in rows]
rows = [row.split("\t") for row in rows]

col_names = {
    0: "source",
    1: "url",
    2: "title",
    3: "image",
    4: "category",
    5: "description",
    6: "rank",
    7: "pubdate",
    8: "video",
}

news = pd.DataFrame(
    rows,
    dtype=str,
)
news = news.rename(columns=col_names)

print("Before pre-processing:")
print(news["category"].value_counts())
print("Category counts:", len(news["category"].value_counts()))
print("Length of df:", len(news))
print("Shape of df:", news.shape)

def pre_process(news: pd.DataFrame):
    # Only keep text content on body_column
    news[body_column] = news[body_column].apply(
        lambda text: "".join(
            [t for t in str(text) if text and t and re.match("[a-zA-Z ]", t)]
        )
    )

    # dup = news.duplicated(subset=["title", "category", body_column], keep="first")
    # news = news[~dup]

    # Remove all duplicates, keep none since duplicates were mostly metadata
    dup = news[body_column].duplicated(keep="first")
    news = news[~dup]

    news[body_column] = news[body_column].str.lower()
    news = news[news[body_column].str.len() > 20]

    # Remove null and empty rows
    news = news[news[body_column].notna() & (news[body_column].str.strip() != "")]

    # news = news[news["category"].str.len() < 30]

    return news


# Pre-process
news = news[col_names.values()] # Remove 9:226

category_vc = news["category"].value_counts()
news = news[news["category"].map(category_vc) > 5]

news = news[news["category"] != "none"]

news[body_column] = news[body_column].str.replace("<[^>]*>", "", regex=True)
news[body_column] = news[body_column].str.replace(
    "http(s?)://[^ ]*", "", regex=True
)

news = pre_process(news)
# /Pre-process

print("After pre-processing:")
print(news["category"].value_counts())
print("Category counts:", len(news["category"].value_counts()))
print("Length of df:", len(news))
print("Shape of df:", news.shape)

# Dist plot
plt.figure(figsize=(10, 6))
news[category_column].value_counts().plot(kind='barh', color='skyblue')
plt.xlabel('Number of News')
plt.title('Distribution of News in Different Categories')
plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.savefig('distplot.png')

# Shuffle df
news = news.sample(frac=1, random_state=365)

news.to_csv(csv_file)
print("Saved cleaned csv in", csv_file)

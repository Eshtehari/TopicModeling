import os

from gensim.test.utils import datapath


class Settings:
    # AG News Dataset
    pardir = os.path.join("..", "data", "ag_news_dataset")
    orig_csv_file = os.path.join(pardir, "newsSpace")
    csv_file = os.path.join(pardir, "newsSpace_cleaned.csv")
    category_column = "category"
    body_column = "description"
    title_column = "title"
    lda_file = datapath("lda_model.bin")
    dictionary_file = os.path.join(pardir, "train.dict")
    # Had to switch to Python 3.8.18 to install OCTIS, Gensim couldn't load model from Python 3.11.5. Upgraded from gensim 4.2.0 to 4.3.2 to no avail. Had to re-train but Octis didn't work after retrain anyways.
    # lda_file = os.path.join("..", "data", "old_gensim", "lda_model.bin")
    # dictionary_file = os.path.join("..", "data", "old_gensim", "train.dict")
    # lda model also saves dictionary!
    # dictionary_file = datapath("lda_model.bin.id2word")
    predicted_csv_file = os.path.join(pardir, "predicted_news.csv")

    # Pre-trained model
    # lda_file = datapath("lda_3_0_1_model")
    # dictionary_file = datapath("lda_3_0_1_model.id2word")
    # predicted_csv_file = os.path.join(pardir, "predicted_news_pretrained.csv")

    # CNN Dataset
    # WARNING: Does not need cleaning, already cleaned on Phase 1 of IR project
    # pardir = os.path.join("..", "data", "cnn_news")
    # csv_file = os.path.join(pardir, "news_cleaned.csv")
    # category_column = "Class"
    # body_column = "Body"
    # lda_file = datapath("lda_model_cnn.bin")
    # dictionary_file = os.path.join(pardir, "train_cnn.dict")
    # predicted_csv_file = os.path.join(pardir, "predicted_news_cnn.csv")

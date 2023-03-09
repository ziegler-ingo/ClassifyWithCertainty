"""
This preprocessing script follows the dataset generation steps of:
    Yibo Hu and Latifur Khan. 2021. Uncertainty-Aware Reliable Text 
    Classification. In Proceedings of the 27th ACM SIGKDD Conference 
    on Knowledge Discovery and Data Mining (KDD ’21), August 14–18, 
    2021, Virtual Event, Singapore. ACM, New York, NY, USA, 9 pages. 
    https://doi.org/10.1145/3447548. 3467382
    https://github.com/snowood1/BERT-ENN/blob/main/prepare_data.py 

The above paper itself follows the procedure of:
    Hendrycks, Dan, Mantas Mazeika, and Thomas Dietterich. 
    "Deep anomaly detection with outlier exposure." 
    arXiv preprint arXiv:1812.04606 (2018).
    https://github.com/hendrycks/outlier-exposure/tree/master/NLP_classification
"""

import os
import sys
import pandas as pd
from datasets import load_dataset

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def format_trec(file):
    lbls, texts = [], []

    with open(file, "rb") as f:
        for line in f:
            # there is a non-ASCII byte, which gets replace with whitespace
            # see: see https://pytorch.org/text/_modules/torchtext/datasets/trec.html
            lbl, _, text = line.replace(b"\xf0", b" ").decode().partition(" ")
            lbls.append(lbl.strip())
            texts.append(text.strip())
    return lbls, texts


DATASETS = sys.argv[1:]
DATA_PATH = "../../../datasets/"
KNOWN_DATASETS = ["20news", "trec", "sst", "snli", "imdb", "m30k", "wmt16", "yelp"]

unknown_datasets = [ds for ds in DATASETS if ds not in KNOWN_DATASETS]
assert not unknown_datasets, (
    f"found unknkown dataset(s): {unknown_datasets}, "
    f"known datasets: {KNOWN_DATASETS}"
)


for dataset in DATASETS:
    if dataset == KNOWN_DATASETS[0]:
        print(f"Dataset to preprocess: {dataset}")
        # 20news
        _path = "20newsgroups/"
        val_split = 0.8
        train = fetch_20newsgroups(
            data_home="/tmp/20news", subset="train", shuffle=True, random_state=0
        )
        test = fetch_20newsgroups(data_home="/tmp/20news", subset="test", shuffle=False)

        train_len = int(val_split * len(train.data))

        train_df = pd.DataFrame(
            {"seq": train.data[:train_len], "lbl": train.target[:train_len]}
        )
        val_df = pd.DataFrame(
            {"seq": train.data[train_len:], "lbl": train.target[train_len:]}
        )
        test_df = pd.DataFrame({"seq": test.data, "lbl": test.target})

        if not os.path.exists(DATA_PATH + _path):
            os.makedirs(DATA_PATH + _path)

        train_df.to_csv(DATA_PATH + _path + "train.csv", index=False)
        val_df.to_csv(DATA_PATH + _path + "val.csv", index=False)
        test_df.to_csv(DATA_PATH + _path + "test.csv", index=False)
        print("Finished saving 20newsgroups.")

    elif dataset == KNOWN_DATASETS[1]:
        print(f"Dataset to preprocess: {dataset}")
        # trec
        _path = "trec/"
        # has been removed from torchtext datasets after version 0.8.0
        # current stable version as of 01 March 2023 is 0.14.0
        # we therefore download the dataset manually and perform the same
        # operations that torchtext loading did
        # see: https://pytorch.org/text/_modules/torchtext/datasets/trec.html
        train_path = DATA_PATH + _path + "train_5500.label"
        test_path = DATA_PATH + _path + "TREC_10.label"

        train_lbls, train_texts = format_trec(train_path)
        test_lbls, test_texts = format_trec(test_path)

        X_train, X_val, y_train, y_val = train_test_split(
            train_texts, train_lbls, test_size=0.2, random_state=42, stratify=train_lbls
        )

        le = LabelEncoder()
        tr_lbls = le.fit_transform(y_train)
        val_lbls = le.transform(y_val)
        te_lbls = le.transform(test_lbls)

        train_df = pd.DataFrame({"seq": X_train, "lbl": tr_lbls})

        val_df = pd.DataFrame({"seq": X_val, "lbl": val_lbls})

        test_df = pd.DataFrame({"seq": test_texts, "lbl": te_lbls})

        if not os.path.exists(DATA_PATH + _path):
            os.makedirs(DATA_PATH + _path)

        train_df.to_csv(DATA_PATH + _path + "train.csv", index=False)
        val_df.to_csv(DATA_PATH + _path + "val.csv", index=False)
        test_df.to_csv(DATA_PATH + _path + "test.csv", index=False)
        print("Finished saving trec.")

    elif dataset == KNOWN_DATASETS[2]:
        print(f"Dataset to preprocess: {dataset}")
        # sst
        # the files are provided via google drive link in README
        # https://github.com/snowood1/BERT-ENN/blob/main/README.md
        # https://github.com/snowood1/BERT-ENN/blob/main/utils.py#L89
        _path = "sst/"

        train_df = pd.read_csv(DATA_PATH + _path + "train.tsv", sep="\t", header=0)
        train_df = train_df.groupby("label").sample(10000)

        val_df = pd.read_csv(DATA_PATH + _path + "dev.tsv", sep="\t", header=0)
        test_df = pd.read_csv(
            DATA_PATH + _path + "sst-test.tsv",
            sep="\t",
            header=None,
            names=["sentence", "label"],
        )
        test_df["sentence"] = test_df["sentence"].str.strip("\n")

        train_df.to_csv(DATA_PATH + _path + "train.csv", index=False)
        val_df.to_csv(DATA_PATH + _path + "val.csv", index=False)
        test_df.to_csv(DATA_PATH + _path + "test.csv", index=False)
        print("Finished saving sst.")

    elif dataset == KNOWN_DATASETS[3]:
        print(f"Dataset to preprocess: {dataset}")
        # snli
        _path = "snli/"
        snli = load_dataset("snli", split="test", data_dir="/tmp/snli")

        test_df = pd.DataFrame({"seq": snli["hypothesis"], "lbl": snli["label"]})

        if not os.path.exists(DATA_PATH + _path):
            os.makedirs(DATA_PATH + _path)

        test_df.to_csv(DATA_PATH + _path + "test.csv", index=False)
        print("Finished saving snli.")

    elif dataset == KNOWN_DATASETS[4]:
        print(f"Dataset to preprocess: {dataset}")
        # imdb
        _path = "imdb/"
        imdb = load_dataset("imdb", split="test", data_dir="/tmp/imdb")

        test_df = pd.DataFrame({"seq": imdb["text"], "lbl": imdb["label"]})

        test_df = test_df.sample(5000)

        if not os.path.exists(DATA_PATH + _path):
            os.makedirs(DATA_PATH + _path)

        test_df.to_csv(DATA_PATH + _path + "test.csv", index=False)
        print("Finished saving imdb.")

    elif dataset == KNOWN_DATASETS[5]:
        print(f"Dataset to preprocess: {dataset}")
        # m30k
        # the files are provided via google drive link in README
        # https://github.com/snowood1/BERT-ENN/blob/main/README.md
        # https://github.com/snowood1/BERT-ENN/blob/main/utils.py#L166
        _path = "multi30k/"

        # yes, the preprocessing of Hu & Khan use train.txt here
        test_df = pd.read_csv(
            DATA_PATH + _path + "train.txt", header=None, names=["text"]
        )
        test_df["label"] = 0

        if not os.path.exists(DATA_PATH + _path):
            os.makedirs(DATA_PATH + _path)

        test_df.to_csv(DATA_PATH + _path + "test.csv", index=False)
        print("Finished saving multi30k.")

    elif dataset == KNOWN_DATASETS[6]:
        print(f"Dataset to preprocess: {dataset}")
        # wmt16
        # the files are provided via google drive link in README
        # https://github.com/snowood1/BERT-ENN/blob/main/README.md
        # https://github.com/snowood1/BERT-ENN/blob/main/utils.py#L187
        _path = "wmt16/"

        test_df = pd.read_table(
            DATA_PATH + _path + "wmt16_sentences", header=None, names=["text"]
        )
        test_df["label"] = 0

        if not os.path.exists(DATA_PATH + _path):
            os.makedirs(DATA_PATH + _path)

        test_df.to_csv(DATA_PATH + _path + "test.csv", index=False)
        print("Finished saving wmt16.")

    elif dataset == KNOWN_DATASETS[7]:
        print(f"Dataset to preprocess: {dataset}")
        # yelp
        # the files are provided via google drive link in README
        # https://github.com/snowood1/BERT-ENN/blob/main/README.md
        # https://github.com/snowood1/BERT-ENN/blob/main/utils.py#L208
        _path = "yelp/"

        test_df = pd.read_csv(
            DATA_PATH + _path + "test_old.csv", header=None, names=["label", "text"]
        )
        test_df = test_df[["text", "label"]]

        if not os.path.exists(DATA_PATH + _path):
            os.makedirs(DATA_PATH + _path)

        test_df.to_csv(DATA_PATH + _path + "test.csv", index=False)
        print("Finished saving yelp.")

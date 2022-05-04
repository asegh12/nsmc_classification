"""
원본 데이터(ratings_test.txt, ratings_train.txt) 필요
"""

from tqdm import tqdm
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def pre_processing(filename):
    reviews = []
    PNlist = []
    with open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines[1:]:
            tmp = '_' + re.sub(r"[^가-힣 ]", "", line.split('\t')[1])
            append_line = re.sub(r" ", "_", tmp)
            reviews.append(append_line)
            PNlist.append(int(line.split('\t')[2]))
    return reviews, PNlist


def make_tfidf_doc(filename, data, label):
    with open(filename, "w") as f:
        for PN, x in zip(label, data):
            f.write(f"{-1 if PN == 0 else PN} ")
            tmplist = list(
                zip([z+1 for z in x.indices.tolist()], x.data.tolist()))
            for i, d in sorted(tmplist, key=lambda x: x[0]):
                f.write(f"{i}:{d} ")
            f.write(f"\n")


train_reviews, train_PN = pre_processing("ratings_train.txt")
test_reviews, test_PN = pre_processing("ratings_test.txt")

vectorizer = TfidfVectorizer(ngram_range=(2, 2), analyzer="char", min_df=3)

train_set = vectorizer.fit_transform(train_reviews)
make_tfidf_doc("tfidf_train.txt", train_set, train_PN)

test_set = vectorizer.transform(test_reviews)
make_tfidf_doc("tfidf_test.txt", test_set, test_PN)

with open("words.txt", 'w', encoding='utf-8') as f:
    for word in vectorizer.get_feature_names_out():
        f.write(f"{word}\n")

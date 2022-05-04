from tqdm import tqdm
import re
from gensim.models import word2vec
import numpy as np

def pre_processing(filename):
    reviews = []
    PNlist = []
    with open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines[1:]:
            # 정규표현식을 이용하여 의미있는 한글 토큰들만 추출
            tmp = '_' + re.sub(r"[^가-힣 ]", "", line.split('\t')[1])
            if len(tmp) < 1 : continue # 추출 후 빈 문장이면 스킵
            append_line = re.sub(r" ", "_", tmp) # 띄어쓰기 -> '_'
            
            # 문장을 바이그램으로 토큰화
            bigram_line = []
            for i in range(len(append_line)-1):
                bigram_line.append(append_line[i:i+2])
            reviews.append(bigram_line)
            
            # 라벨 추출
            PNlist.append(-1 if int(line.split('\t')[2])==0 else 1)
    return reviews, PNlist


def get_dataset(reviews, model, num_features):
    dataset = list()
    Nulllist = []

    for s, l in zip(reviews, range(len(reviews))) :
        # 출력 벡터 초기화
        feature_vector = np.zeros((num_features), dtype=np.float32)
        num_words = 0
        # 어휘사전 준비
        index2word_set = set(model.wv.index_to_key)

        for w in s:
            # 사전에 해당하는 단어에 대해 단어 벡터를 더함
            if w in index2word_set:
                num_words +=1
                feature_vector = np.add(feature_vector, model.wv[w])

        if num_words==0 : 
            Nulllist.append(l)
            continue
        
        # 문장의 단어 수만큼 나누어 단어 벡터의 평균값을 문장 벡터로 함
        feature_vector = np.divide(feature_vector, num_words)
        dataset.append(feature_vector)

    reviewFeatureVecs = np.stack(dataset)
    return reviewFeatureVecs, Nulllist


train_reviews, train_PN = pre_processing("ratings_train.txt")
test_reviews, test_PN = pre_processing("ratings_test.txt")

# 워드임베딩 모델 만들기
model = word2vec.Word2Vec(train_reviews, workers=4, vector_size=100, min_count=3, sample = 1e-3)

# 모델으로 훈련, 검증 집합의 각 문장 벡터들의 합, 라벨 가져오기(get_dataset)
train_data_vecs, null_train = get_dataset(train_reviews, model, 100)
test_data_vecs, null_test = get_dataset(test_reviews, model, 100)

# 문장 벡터들의 합을 만드는 과정중 문제 있는 라벨들을 삭제
for i in sorted(null_train, key=lambda x : -x) : del train_PN[i]
for i in sorted(null_test, key=lambda x : -x) : del test_PN[i]

from sklearn import svm
sv = svm.SVC(gamma='scale')
sv.fit(train_data_vecs, train_PN)
print("SVM Accuracy: {}".format(sv.score(test_data_vecs, test_PN)))

from sklearn.linear_model import LogisticRegression
lgs = LogisticRegression(class_weight = 'balanced', max_iter=500, n_jobs=4)
lgs.fit(train_data_vecs, train_PN)
print("LogisticRegression Accuracy: {}".format(lgs.score(test_data_vecs, test_PN)))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=4)
clf.fit(train_data_vecs, train_PN)
print("RandomForest Accuracy: {}".format(clf.score(test_data_vecs, test_PN)))

from sklearn import tree 
df = tree.DecisionTreeClassifier() 
df.fit(train_data_vecs, train_PN)
print("DecisionTree Accuracy: {}".format(df.score(test_data_vecs, test_PN)))

from sklearn.neural_network import MLPClassifier 
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000) 
mlp.fit(train_data_vecs, train_PN)
print("MLP Accuracy: {}".format(mlp.score(test_data_vecs, test_PN)))
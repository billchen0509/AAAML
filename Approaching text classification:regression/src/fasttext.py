# fasttext.py

import io
import numpy as np 
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

def load_vectors(fname):
    fin = io.open(
        fname,
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )

    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def sentence_to_vec(s, embedding_dict, stop_words, tokenizer):
    words = str(s).lower()
    words = tokenizer(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        if w in embedding_dict:
            M.append(embedding_dict[w])
    if len(M) == 0:
        return np.zeros(300)
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

if __name__ == '__main__':
    df = pd.read_csv('../input/imdb.csv')

    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == 'positive' else 0
    )

    # the next step is to randomize the rows of the data
    df = df.sample(frac = 1).reset_index(drop = True)

    # load embeddings into memory
    print('Loading embeddings')
    embedding = load_vectors('../input/crawl-300d-2M.vec')

    # create sentence embeddings
    print('Creating sentence vectors')
    vectors = []

    for review in df.review.values:
        vectors.append(
            sentence_to_vec(
                s = review,
                embedding_dict = embedding,
                stop_words = [],
                tokenizer = word_tokenize
            )
        )
    vectors = np.array(vectors)

    # fetch labels
    y = df.sentiment.values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits = 5)

    for fold_, (t_, v_) in enumerate(kf.split(X = vectors, y = y)):
        print(f'Fold: {fold_}')
        xtrain = vectors[t_, :]
        ytrain = y[t_]
        xtest = vectors[v_, :]
        ytest = y[v_]

        model = linear_model.LogisticRegression()
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)

        accuracy = metrics.accuracy_score(ytest, preds)
        print(f'Fold: {fold_}')
        print(f'Accuracy = {accuracy}')
        print('')





# pipeline_search.py

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def quadratic_weighted_kappa(y_true, y_pred):
    return metrics.cohen_kappa_score(y_true, y_pred, weights='quadratic')

if __name__ == '__main__':

    # load the data
    train = pd.read_csv('../input/train.csv')

    # we dont need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)

    # create labels.drop useless columns
    y = train.relevance.values

    # do some lambda magic on text columns
    traindata = list(
        train.apply(lambda x:'%s %s' % (x['text1'], x['text2']),axis=1)
    )
    testdata = list(
        test.apply(lambda x:'%s %s' % (x['text1'], x['text2']),axis=1)
    )

    # tfidf vecorizer
    tfv = TfidfVectorizer(
        min_df=3,
        max_features=None,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 3),
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1,
        stop_words='english'
    )

    # Fit TFIDF
    tfv.fit(traindata)
    X = tfv.transform(traindata)
    X_test = tfv.transform(testdata)

    # Initialize SVD
    svd = TruncatedSVD()

    # Initialize the standard scaler
    scl = StandardScaler()

    # We will use SVM here..
    svm_model = SVC()

    # create pipeline
    clf = pipeline.Pipeline(
        [
            ('svd', svd),
            ('scl', scl),
            ('svm', svm_model)
        ]
    )

    # create parameter grid
    param_grid = {
        'svd__n_components': [300, 400],
        'svm__C': [10, 12]
    }

    # kappa scorer
    kappa_scorer = metrics.make_scorer(
        quadratic_weighted_kappa,
        greater_is_better=True
    )

    # Initialize Grid Search Model
    model = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring=kappa_scorer,
        verbose=10,
        n_jobs=-1,
        refit=True,
        cv=5
    )

    # Fit Grid Search Model
    model.fit(X, y)
    print('Best score: %0.3f' % model.best_score_)
    print('Best parameters set:')
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f'\t{param_name}: {best_parameters[param_name]}')

    # get best model
    best_model = model.best_estimator_

    # fit model with best parameters
    best_model.fit(X, y)
    preds = best_model.predict(X_test)
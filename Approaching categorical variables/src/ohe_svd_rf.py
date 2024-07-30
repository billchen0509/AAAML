# ohe_svd_rf.py

import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import decomposition
from sklearn import preprocessing
from scipy import sparse

def run(fold):
    df = pd.read_csv('../input/cat_train_folds.csv')
    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    # initialize OneHotEncoder from scikit-learn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    full_data = pd.concat([df_train[features], df_valid[features]], axis = 0)
    ohe.fit(full_data[features])

    # transform training/validation data
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    # initialize Truncated SVD
    svd = decomposition.TruncatedSVD(n_components = 120)
    # fit svd on full sparse data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    # transform x_train and x_valid
    x_train = svd.transform(x_train)
    x_valid = svd.transform(x_valid)

    # initialize Random Forest model
    model = ensemble.RandomForestClassifier(n_jobs = -1)
    model.fit(x_train, df_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

# lbl_xgb_num_feat.py

import itertools
import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

def feature_engineering(df, cat_cols):
    '''
    This function is used for feature engineering
    '''
    # this will create all 2-combinations of values
    # in this list
    # for example:
    # list(itertools.combinations([1,2,3], 2)) will return
    # [(1,2), (1,3), (2,3)]
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:, c1 + '_' + c2] = df[c1].astype(str) + '_' + df[c2].astype(str)
    return df

def run(fold):
    df = pd.read_csv('../input/adult_folds.csv')

    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    target_mapping = {
        '<=50K': 0,
        '>50K': 1
    }
    df.loc[:, 'income'] = df.income.map(target_mapping)

    # list of categorical columns for feature engineering
    cat_cols = [c for c in df.columns if c not in num_cols and c not in ('kfold', 'income')]

    # add new features
    df = feature_engineering(df, cat_cols)

    # all columns are features except kfold and income columns
    features = [f for f in df.columns if f not in ('kfold', 'income')]

    # fill all NaN values with NONE
    for col in features:
        # do not fill numerical columns with 'NONE'
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna('NONE')

    # label encode all the features
    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])

    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(n_jobs = -1,max_depth=7, n_estimators=200)
    model.fit(x_train, df_train.income.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
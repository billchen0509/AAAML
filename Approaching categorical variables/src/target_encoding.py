# target_encoding.py
import copy
import pandas as pd

from sklearn import metrics,preprocessing
import xgboost as xgb

def mean_target_encoding(data):
    df = copy.deepcopy(data)
    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    # map targets to 0s and 1s
    target_mapping = {
        '<=50K': 0,
        '>50K': 1
    }
    df.loc[:, 'income'] = df.income.map(target_mapping)
    # all columns are features except kfold and income columns
    features = [f for f in df.columns if f not in ('kfold', 'income') and f not in num_cols]

    # fill all NaN values with NONE
    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna('NONE')
    
    # label encode all the features
    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])
    
    # a list to store 5 validation dataframes
    encoded_dfs = []

    # go over all folds
    for fold in range(5):
        # fetch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop = True)
        df_valid = df[df.kfold == fold].reset_index(drop = True)
        
        # for all categorical columns
        for column in features:
            mapping_dict = dict(
                df_train.groupby(column)['income'].mean()
            )
            df_valid.loc[:, column + '_enc'] = df_valid[column].map(mapping_dict)
        # add this dataframe to the list
        encoded_dfs.append(df_valid)
    # create full data frame again and return
    encoded_df = pd.concat(encoded_dfs, axis = 0)
    return encoded_df

def run(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    features = [f for f in df.columns if f not in ('kfold', 'income')]
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(n_jobs = -1, max_depth = 7)
    model.fit(x_train, df_train.income.values)

    # predict on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    df = pd.read_csv('../input/adult_folds.csv')
    df = mean_target_encoding(df)
    for fold_ in range(5):
        run(df, fold_)
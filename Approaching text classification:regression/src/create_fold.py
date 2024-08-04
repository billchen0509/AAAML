# create_fold.py

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/imdb.csv")

    df.sentiment = df.sentiment.apply(lambda x:1 if x == 'positive' else 0)

    df['kfold'] = -1
    df = df.sample(frac = 1).reset_index(drop = True)

    y = df.sentiment.values

    kf = model_selection.StratifiedKFold(n_splits = 5)
    
    for f, (t_, v_) in enumerate(kf.split(X = df, y = y)):
        df.loc[v_, 'kfold'] = f
    
    df.to_csv("../input/imdb_folds.csv", index = False)
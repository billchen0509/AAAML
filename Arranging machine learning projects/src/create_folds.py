# src/create_folds.py

import pandas as pd
from sklearn import model_selection

data = pd.read_csv("../input/mnist_train.csv")

def create_folds(data):
    data['kfold'] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    y = data['label'].values
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=y)):
        data.loc[v_, 'kfold'] = f
    return data

if __name__ == "__main__":
    df = create_folds(data)
    df.to_csv("../input/mnist_train_folds.csv", index=False)

# rf_hyperopt.py
import numpy as np 
import pandas as pd

from functools import partial

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimize(params, x ,y):

    
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)

    # initialize accuracy list
    accuracies = []

    # loop over all folds
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        # fit model for current fold
        model.fit(xtrain, ytrain)

        #create predictions
        preds = model.predict(xtest)

        # calculate and append accuracy
        fold_accuracy = metrics.accuracy_score(
            ytest,
            preds
        )
        accuracies.append(fold_accuracy)
    return -1 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_train.csv")

    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    param_space = {

        "max_depth": scope.int(hp.quniform("max_depth", 1, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 1)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0.01, 1)
    }

    # partial function
    optimization_function = partial(
        optimize,
        x=X,
        y=y
    )

    # initialize trials to keep logging information
    trials = Trials()

    # run hyperopt
    hopt = fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials
    )
    print(hopt)
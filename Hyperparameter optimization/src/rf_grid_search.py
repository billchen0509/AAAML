# rf_grid_search.py

import numpy as np 
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    # define the model here
    # n_jobs=-1 => use all cores
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    # define a grid of parameters
    param_grid = {
        'n_estimators': [100, 200, 250, 300, 400, 500],
        'max_depth': [1, 2, 5, 7, 11, 15],
        'criterion': ['gini', 'entropy']
    }

    # initialize grid search
    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=10,
        n_jobs=1,
        cv=5 # cross-validation 5 folds
    )

    # fit the model and extract best score
    model.fit(X, y)
    print(f"Best score: {model.best_score_}")

    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
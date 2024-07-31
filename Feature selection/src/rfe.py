# rfe.py

import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data['data']
col_names = data['feature_names']
y = data['target']

# initialize the model
model = LinearRegression()

# initialize RFE
rfe = RFE(
    estimator=model,
    n_features_to_select=3
)

# fit RFE
rfe.fit(X, y)

# get the transformed data with selected columns
X_transformed = rfe.transform(X)
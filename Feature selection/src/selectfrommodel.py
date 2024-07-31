# selectfrommodel.py
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# fetch a regression dataset
# in diabetes data we predict diabetes progression # after one year based on some features
data = load_diabetes()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]

# initialize the model
model = RandomForestRegressor()

# select from the model
sfm = SelectFromModel(estimator=model) 
X_transformed = sfm.fit_transform(X, y)

# see which features were selected
support = sfm.get_support()
# get feature names
print([
x for x, y in zip(col_names, support) if y == True
])
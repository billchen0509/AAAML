# greedy.py
import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:
    """
    A simple and custom class for greedy feature selection.
    """
    def evaluate_score(self, X, y):
        """
        This function evaluates model on data and returns Area Under ROC Curve (AUC)
        NOTE: We fit the data and calculate AUC on same data. 
        WE ARE OVERFITTING HERE. 
        But this is also a way to achieve greedy selection.
        """
        # build a classification model
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc
    
    def _feature_selection(self, X, y):
        """
        This function does the actual greedy selection
        """
        good_features = []
        best_score = 0
        
        # calculate the number of features
        num_features = X.shape[1]

        # infinite loop
        while True:
            # initialize best feature and score of this loop
            this_feature = None
            best_score_this_time = 0
            
            # loop over all features
            for feature in range(num_features):
                # if feature is already in good features, skip this for loop
                if feature in good_features:
                    continue
                
                # selected features
                selected_features = good_features + [feature]
                
                
                # remove all other features from data
                xtrain = X[:, selected_features]
                # evaluate score
                score = self.evaluate_score(xtrain, y)
                
                # if score is greater than the best score of this loop
                # then set this feature as best feature and update best score
                if score > best_score_this_time:
                    this_feature = feature
                    best_score_this_time = score
            
            # if we have selected a feature, add it to good feature
            # and update best score of all loops
            if this_feature != None:
                good_features.append(this_feature)
                best_score = best_score_this_time
            
            # if we didnt improve the score this time, exit the loop
            if best_score_this_time < best_score:
                break
        
        return good_features, best_score
    
    def __call__(self, X, y):
        """
        Call function will call the class on a set of arguments
        """
        # select features, return scores and selected indices
        scores, features = self._feature_selection(X, y)
        return X[:, features], scores

if __name__ == '__main__':
    # generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=100)
    
    # transform the data
    X_transformed, scores = GreedyFeatureSelection()(X, y)
    
    # print results
    print(scores)
    print(X_transformed)
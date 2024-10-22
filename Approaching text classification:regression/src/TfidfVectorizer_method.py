# import what we need
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams

if __name__ == "__main__":
    # read the training file
    df = pd.read_csv("../input/imdb.csv")

    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )

    # we create a new column called kfold and fill it with -1
    df['kfold'] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac = 1).reset_index(drop = True)

    # fetch labels
    y = df.sentiment.values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits = 5)

    # fill the new kfold column
    for f , (t_, v_) in enumerate(kf.split(X = df, y = y)):
        df.loc[v_, 'kfold'] = f
    
    for fold_ in range(5):
        train_df = df[df.kfold != fold_].reset_index(drop = True)
        test_df = df[df.kfold == fold_].reset_index(drop = True)

        # initialize TfidfVectorizer with NLTK's word_tokenize
        tfidf_vec = TfidfVectorizer(
            tokenizer = word_tokenize,
            token_pattern = None
        )
        # tfidf_vec = TfidfVectorizer(
        #     tokenizer = word_tokenize,
        #     token_pattern = None,
        #     ngram_range = (1, 3)
        # )
        tfidf_vec.fit(train_df.review)

        # transform the training and validation data
        xtrain = tfidf_vec.transform(train_df.review)
        xtest = tfidf_vec.transform(test_df.review)

    
        # initialize logistic regression model
        model = linear_model.LogisticRegression()

        # fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)

        preds = model.predict(xtest)

        accuracy = metrics.accuracy_score(test_df.sentiment, preds)

        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
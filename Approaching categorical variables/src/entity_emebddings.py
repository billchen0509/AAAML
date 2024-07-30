# entity_embeddings.py
import os
import gc
import joblib
import pandas as pd
import numpy as np

from sklearn import metrics, preprocessing
from tensorflow.keras import layers, optimizers, callbacks, utils
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

def create_model(data, catcols):
    inputs = []
    outputs = []

    # loop over all categorical columns
    for c in catcols:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

        # simple keras input layer with size 1
        inp = layers.Input(shape = (1,))

        # add embedding layer to raw input
        out = layers.Embedding(num_unique_values + 1, embed_dim, name = c)(inp)

        # 1-d spatial dropout is the standard for embedding layers
        out = layers.SpatialDropout1D(0.3)(out)

        # reshape the input to the dimension of embedding
        out = layers.Reshape(target_shape = (embed_dim, ))(out)

        inputs.append(inp)
        outputs.append(out)

    # concatenate all output layers
    x = layers.Concatenate()(outputs)

    # add a batchnorm layer
    x = layers.BatchNormalization()(x)

    # a bunch of dense layers with dropout.
    x = layers.Dense(300, activation = 'relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation = 'relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)        
    # using softmax as this is a binary classification problem
    y = layers.Dense(2, activation = 'softmax')(x)

    # create final model
    model = Model(inputs = inputs, outputs = y)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    return model
    
def run(fold):
    df = pd.read_csv('../input/cat_train_folds.csv')

    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')

    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df.loc[:, feat] = lbl_enc.fit_transform(df[feat].values)

    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    model = create_model(df, features)
    xtrain = [df_train[feat].values for feat in features]
    xvalid = [df_valid[feat].values for feat in features]

    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    # convert target columns to categories
    y_train_cat = utils.to_categorical(ytrain)
    y_valid_cat = utils.to_categorical(yvalid)

    model.fit(xtrain, y_train_cat, validation_data = (xvalid, y_valid_cat), verbose = 1, batch_size = 1024, epochs = 3)
    valid_preds = model.predict(xvalid)[:, 1]

    auc = metrics.roc_auc_score(yvalid, valid_preds)
    print(f'Fold = {fold}, AUC = {auc}')

    K.clear_session()

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
# Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
import glob

import math
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.layers import LeakyReLU, ELU, Activation, Reshape, Flatten, Conv2D, MaxPooling2D, Conv1D, Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.models import model_from_json


# Utility function to load input data
def load_cnninput1(path):
    filenames = glob.glob(path + "/*.csv")

    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename, index_col=0))

    # from original format to matrix as an input.
    # in order to perform CNN we need to have a dim of 60 * 60
    # to be able to split the data in equal parts so we als o increased the shape to 80 * 80

    keras_input = []
    for df in range(0, len(dfs)):
        pivot_df = dfs[df].pivot_table(index='bin_y', columns='bin_x', values='freq', fill_value=0).rename_axis(None).to_numpy()
        # from 59*59 to a 60*60 dimension by adding a column with 0s at the end and a row with 0's below
        pivot_df = np.pad(pivot_df, [(0, 1), (0, 1)], mode='constant')
        keras_input.append(pivot_df)

    x_1 = np.concatenate(keras_input)
    X = np.array(x_1).reshape(-1, 60, 60, 1)
    # reshape input for CNN
    # The input is 60*60 of 100 samples

    labels = []
    for i in range(0, len(dfs)):
        value = dfs[i]['name'].unique()
        labels.append(value[~pd.isnull(value)])

    # y = pd.factorize(labels)[0]
    return(X, labels)

# Training data X_1 monotherapy
path = r'/data/training/'  # add your full path
X_1, labels1 = load_cnninput1(path)
labels1 = pd.DataFrame(labels1)

# Test data X_combo
# combination therapy cases n =<50 stacked 100 times randomly
# combination therapy stacked n = 50
pathc = r'/data/test/combi/'  # add your full path
X_combo, labels_combo = load_cnninput1(pathc)
labels_combo = pd.DataFrame(labels_combo)

# Test data X_combobt
# combination therapy in FAERS but only the drug combinations seen in benchmark data
# combination therapy in bench
pathct = r'/data/test/combi_bt/'  # add your full path
X_combobt, labels_combobt = load_cnninput1(pathct)
labels_combobt = pd.DataFrame(labels_combobt)

# Test data X_bt
# monotherapy in FAERS but only the drugs seen in benchmark data
pathbt = r'/data/test/mono/'  # add your full path
X_bt, labels_bt = load_cnninput1(pathbt)
labels_bt = pd.DataFrame(labels_bt)

# Modelling the CNN
# input data shape of your data
inp = Input((60, 60, 1))

# ENCODER
# first layer of the CNN  padding keeps the shape the same
e = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
e = MaxPooling2D((2, 2))(e)

# second layer of the CNN
e = Conv2D(32, (3, 3), activation='relu', padding='same')(e)
e = MaxPooling2D((2, 2))(e)

# 3th layer of the CNN
e = Conv2D(32, (3, 3), activation='relu', padding='same')(e)

# Latent space // bottleneck layer
l = Flatten()(e)
l = Dense(15*15, activation='softmax')(l)
z = ELU()(l)

# DECODER
# needs to be exactly the same layers as the encoder
# output shape should be same as input
# reshape dense layer in order to decode the output
d = Reshape((15, 15, 1))(l)

d = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)

d = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)

d = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(d)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

# MODEL 1: ENCODER
encoder = Model(inp, z)
encoder.summary()

# MODEL 2: AUTOENCODER / DECODER
ae = Model(inp, decoded)
ae.summary()

# Running the CNN model
np.random.seed(123)
ae.compile(loss="mse",
           optimizer="adam",
           metrics=["accuracy"])
ae.fit(X_1, X_1, epochs=10)


# Predictions
pred_mono = ae.predict(X_1, verbose=1)
pred_combo = ae.predict(X_combo, verbose=1)
pred_combobt = ae.predict(X_combobt, verbose=1)
pred_monobt = ae.predict(X_bt, verbose=1)


# Performance evaluation
mono_acc = ae.evaluate(X_1, X_1, verbose=1)
combo_acc = ae.evaluate(X_combo, X_combo, verbose=1)
combobt_acc = ae.evaluate(X_combobt, X_combobt, verbose=1)
monobt_acc = ae.evaluate(X_bt, X_bt, verbose=1)


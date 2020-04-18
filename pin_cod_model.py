#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" SEMEVAL 2020 Task 12: OffensEval 2020
    Sub-task A - Offensive language  identification for Turkish Language
	This script describes the system of "pin_cod_".
"""

from pin_cod_preprocessing import preprocessed_training_for_neural_network, preprocessed_testset_for_neural_network, to_categorical_labels, X_features_dense, Xtest_features_dense, testset_dataset_features_labels
import fasttext
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report
import random as rn

# The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
np.random.seed(42)

# The below is necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(12345)

""" TRAINING SET """
# X contains 31756 tweets
X = preprocessed_training_for_neural_network()
y_labels, labels = to_categorical_labels() # use labels for neural network model
X_4_dense, X_5_dense, X_6_dense, X_11_dense, X_12_dense, X_14_dense, X_16_dense, X_17_dense, X_19_dense, X_21_dense, X_24_dense = X_features_dense()

""" TESTSET """
# Xtest contains 3528 tweets
Xtest = preprocessed_testset_for_neural_network()
_, _, Xtest_ids = testset_dataset_features_labels()
Xtest_0_dense, Xtest_1_dense, Xtest_2_dense, Xtest_3_dense, Xtest_4_dense, Xtest_5_dense, Xtest_6_dense, Xtest_7_dense, Xtest_8_dense, Xtest_9_dense, Xtest_11_dense = Xtest_features_dense()

  
#index all words in X (i.e., training set)
from collections import defaultdict
label_count = defaultdict(int)
for indx_t, tweet in enumerate(X):
    for indx_w, word in enumerate(tweet):
        label_count[word] += 1

word_index_all = {} # X (i.e., training set) contains 98115 unique words
for i, w in enumerate(label_count, start=1):
    word_index_all[w] = i
    
def get_word_index_input(data, word_index_all):
    sequences = []
    for x in data:
        seq_per_tweet = []
        for w in x:
            if w in word_index_all:
                seq_per_tweet.append(word_index_all[w])
        sequences.append(seq_per_tweet)
        
    # find the max word length per tweet
    max_word_length = []
    for s in sequences:
        max_word_length.append(len(s))
    max_len = max(max_word_length)
    
    word_index = {}
    for x in data:
        for w in x:
            if w in word_index_all:
                word_index[w] = word_index_all[w]
    
    # pad sequences
    input1 = pad_sequences(sequences, maxlen = max_len)
    
    return word_index, input1, max_len

# textual_input
word_index1, input1, max_len = get_word_index_input(X, word_index_all) # max_len: 82 in the training set
word_index_test, input_test, max_len_test = get_word_index_input(Xtest, word_index_all) # max_len: 57 in the test set

# for training set
inputs_embeddings = [input1]
all_inputs = inputs_embeddings

word_index1_list = [word_index1]
concatenated_word_indices = dict(list(word_index1.items()))

all_inputs.append(X_4_dense)
all_inputs.append(X_5_dense)
all_inputs.append(X_6_dense)
all_inputs.append(X_11_dense)
all_inputs.append(X_12_dense)
all_inputs.append(X_14_dense)
all_inputs.append(X_16_dense)
all_inputs.append(X_17_dense)
all_inputs.append(X_19_dense)
all_inputs.append(X_21_dense)
all_inputs.append(X_24_dense)

# split the provided training dataset into 2 parts: (1) a training set(80%) and (2) a validation set(20%)
input_training= []
input_validation= []

labels_training= []
labels_validation= []

for i in all_inputs:
    X_train1, X_val1, y_train, y_val = train_test_split(i, labels, test_size=0.20, random_state=7)
    input_training.append(X_train1)
    input_validation.append(X_val1)

labels_training = y_train
labels_validation = y_val

# for the testset
final_testset = [input_test,Xtest_0_dense, Xtest_1_dense, Xtest_2_dense, Xtest_3_dense, Xtest_4_dense, Xtest_5_dense, Xtest_6_dense, Xtest_7_dense, Xtest_8_dense, Xtest_9_dense, Xtest_11_dense]

# Load fasttext pre-trained word embedding model
model = fasttext.load_model("cc.tr.300.bin")

vector_dim = 300 
embedding_matrix = np.zeros((len(concatenated_word_indices) + 1, vector_dim))
        
for word, i in concatenated_word_indices.items():
    try:
        embedding_vector = model[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        pass

# Create the model
all_inputs_keras =[]

current_input = Input(shape=(None,)) 
word_embeddings = Embedding(output_dim = vector_dim, 
               input_dim = len(concatenated_word_indices) + 1, 
               weights = [embedding_matrix],
               input_length = 82, 
               trainable=False)(current_input)

all_inputs_keras.append(current_input)

# sequences of word embeddings are fed to RNN (bidirection long short term memory networks)
recurrent_output = Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))(word_embeddings)

X_4_dense_input = Input(shape=(X_4_dense.shape[1],))
X_5_dense_input = Input(shape=(X_5_dense.shape[1],))
X_6_dense_input = Input(shape=(X_6_dense.shape[1],))
X_11_dense_input = Input(shape=(X_11_dense.shape[1],))
X_12_dense_input = Input(shape=(X_12_dense.shape[1],))
X_14_dense_input = Input(shape=(X_14_dense.shape[1],))
X_16_dense_input = Input(shape=(X_16_dense.shape[1],))
X_17_dense_input = Input(shape=(X_17_dense.shape[1],))
X_19_dense_input = Input(shape=(X_19_dense.shape[1],))
X_21_dense_input = Input(shape=(X_21_dense.shape[1],))
X_24_dense_input = Input(shape=(X_24_dense.shape[1],))

all_inputs_keras.append(X_4_dense_input)
all_inputs_keras.append(X_5_dense_input)
all_inputs_keras.append(X_6_dense_input)
all_inputs_keras.append(X_11_dense_input)
all_inputs_keras.append(X_12_dense_input)
all_inputs_keras.append(X_14_dense_input)
all_inputs_keras.append(X_16_dense_input)
all_inputs_keras.append(X_17_dense_input)
all_inputs_keras.append(X_19_dense_input)
all_inputs_keras.append(X_21_dense_input)
all_inputs_keras.append(X_24_dense_input)

# merge flattened input models with bilstm output
concatenated_tweet_lvl = concatenate([recurrent_output, X_4_dense_input, X_5_dense_input, X_6_dense_input, X_11_dense_input, X_12_dense_input, X_14_dense_input, X_16_dense_input, X_17_dense_input, X_19_dense_input, X_21_dense_input, X_24_dense_input])

# interpretation model
hidden1 = Dense(100, activation="relu")(concatenated_tweet_lvl)
hidden2 = Dense(100, activation="relu")(hidden1)
output = Dense(2, activation="sigmoid")(hidden2)

model = Model(inputs = all_inputs_keras, outputs = output)

#summarize layers
print(model.summary())

# compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, verbose=0),
        ModelCheckpoint('the_saved_model_is_here.h5',
                        monitor='val_accuracy', save_best_only=True, verbose=0)
        ]

# fit model
model.fit(x = input_training, y = labels_training, validation_data=(input_validation, labels_validation), epochs = 50, batch_size = 100, callbacks=callbacks, shuffle=False)

"""
# To load the saved model later
from keras.models import load_model
model = load_model('the_saved_model_is_here.h5')
"""

# Predict classes for the testset
Ypredicted = model.predict(final_testset,verbose=0)
Ypredicted_classes = Ypredicted.argmax(axis=-1)

# write a csv file consisting of the predicted labels for the testset
predictions = open("the_predicted_labels_are_here.csv", "w")
for indx, l in enumerate(Ypredicted_classes):
    if l == 0:
        labels = "NOT"
    if l == 1:
        labels = "OFF"
    predictions.write("{},{}\n".format(Xtest_ids[indx],labels))
predictions.close()

# Predict classes for the validation set
Ypredicted_val = model.predict(input_validation,verbose=0)
Ypredicted_classes_val = Ypredicted_val.argmax(axis=-1)

def convert_one_hot_encoded_labels_to_integers(labels):
    #convert one hot encoded labels into integers
    #e.g. 0 1 --> 1
    converted_labels = []
    for l in labels:
        for index, single_label in enumerate(l):
            if single_label == 1.0:
                converted_labels.append(index)
    return converted_labels

# make both true labels and predicted labels the same type "list --> int" (for validation set)
converted_true_labels = convert_one_hot_encoded_labels_to_integers(labels_validation)

# Evaluate the model
print("training set results\n")
scores = model.evaluate(input_training, labels_training, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print("*"*30)
print("Validation set results\n")
scores = model.evaluate(input_validation, labels_validation, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print("*"*30)
print(classification_report(converted_true_labels, Ypredicted_classes_val))

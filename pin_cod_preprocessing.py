#!/usr/bin/env python3
# -*- coding: utf-8 -*-


""" SEMEVAL 2020 Task 12: OffensEval 2020
    Sub-task A - Offensive language  identification for Turkish Language
	This is the preprocessing file used in "pin_cod_" system
"""

import string
import pandas as pd
import numpy as np
from scipy import sparse
from collections import defaultdict
import re
from keras.utils import to_categorical

def read_tsvfile(add_your_filename):
	your_tsvfile = []
	with open(add_your_filename, 'r') as fp:
		head = next(fp)
		for line in fp:
			your_tsvfile.append(line.strip().split('\t'))
	return your_tsvfile

def read_txtfile(add_your_filename):
	your_txtfile = []
	with open(add_your_filename, "r") as txtfile:
		head = next(txtfile)
		for row in txtfile:
			your_txtfile.append(row.strip("\n"))
	return your_txtfile

def get_tweets_and_labels(add_your_data):
	tweets = []
	labels = []
	for tweet in add_your_data:
		tweets.append(tweet[1])
		labels.append(tweet[2])
	return tweets, labels

def get_ids_and_tweets(add_your_test_data):
    testset_ids = []
    tweets = []
    for tweet in add_your_test_data:
        testset_ids.append(tweet[0])
        tweets.append(tweet[1])
    return testset_ids, tweets

def label_distribution(labels):
    #show the label distribution
    label_count = defaultdict(int)
    for l in labels:
        label_count[l] += 1
    return label_count

def hashtag_presence(text_in_list):
	hashtagPresence = []
	for line in text_in_list:
		if "#" in line:
			hashtagPresence.append(1)
		else:
			hashtagPresence.append(0)
	return hashtagPresence #a list showing presence of hashtag per tweet [0,1,0,0,1,...]


def remove_hashtag_symbols(text_in_list):
    #This will delete only # symbols but this keeps text after the hashtag 
    #e.g. #calmdown becomes calmdown
    text_without_hashtag_symbols = [] 
    for line in text_in_list:
            line = line.strip().split()
            line_string = " ".join(line)
            rexp_string = r'#'
            tags_created_rexp = re.compile(rexp_string)
            text_cleaned = tags_created_rexp.sub(" ",line_string)
            text_without_hashtag_symbols.append(text_cleaned)
    return text_without_hashtag_symbols

def add_whitespace_before_and_after_punctuation(data_list_of_list):
    #put a whitespace before and after punctuation
    selected_punctuations = '!\"#$%&()+,_-./:;<=>?[\]^`{|}~'
    text_with_space_before_punctuation = []

    for line in data_list_of_list:
        text_with_space_before_punctuation.append(["".join(line).translate(str.maketrans({key: " {0} ".format(key) for key in selected_punctuations}))])

    tokenized_text_with_space_before_punctuation = []
    for line in text_with_space_before_punctuation:
        tokenized_text_with_space_before_punctuation.append("".join(line).lower().strip().split())
    
    return tokenized_text_with_space_before_punctuation

def user_mention_count(data_tokenized):
	user_mention_tags = []
	for x in data_tokenized:
		user_mentions_per_tweet = []
		for w in x:
			if "@user" in w:
				user_mentions_per_tweet.append(w)
		user_mention_tags.append(user_mentions_per_tweet)

	user_mention_count = []
	for x in user_mention_tags:
		user_mention_count.append(len(x))
	return user_mention_count #a list showing user mention counts for each tweet [0,1,0,0,3,5,...0,,1,0]

def user_mention_presence(user_mention_count):
	user_mention_presence = []
	for x in user_mention_count:
		if x > 0:
			user_mention_presence.append(1)
		else:
			user_mention_presence.append(0)
	return user_mention_presence #a list showing @user presence per tweet [0,0,1,0,1..]


def karaliste_words(data_tokenized, karaliste_wordlist):
    #various wordlists were compiled to create karaliste wordlists (Offensive/Profane/Slang word lists)
	karaliste_words = []
	for x in data_tokenized:
		kara_words_per_tweet = []
		for w in x:
			if w in karaliste_wordlist:
				kara_words_per_tweet.append(w)
		karaliste_words.append(kara_words_per_tweet)
	return karaliste_words

def karaliste_word_count(X_karaliste_words):
    #various wordlists were compiled to create karaliste wordlists (Offensive/Profane/Slang word lists)
	karaliste_word_count = []
	for x in X_karaliste_words:
		karaliste_word_count.append(len(x))
	return karaliste_word_count #a list showing karaliste word counts for each tweet [0,1,0,0,2,0,1,....2,0]

def karaliste_word_presence(karaliste_word_count):
    #various wordlists were compiled to create karaliste wordlists (Offensive/Profane/Slang word lists)
	karaliste_wordPresence = []
	for x in karaliste_word_count:
		if x > 0:
			karaliste_wordPresence.append(1)
		else:
			karaliste_wordPresence.append(0)
	return karaliste_wordPresence # a list showing karaliste words presence per tweet [0,1,1,0,1..]


def count_hurtlex_words(raw_data, hurtlex_category):
	hurtlex_word_count = []
	
	for x in raw_data:
		hurtlex_word_per_tweet = []
		for h in hurtlex_tr:
			if h[1] == hurtlex_category:
				if h[3] in x.lower():
					hurtlex_word_per_tweet.append(h[3])

		hurtlex_word_count.append(len(hurtlex_word_per_tweet))
	return hurtlex_word_count #a list showing hurtlex_word_count for a given hurtlex category [0,1,0,0,...3,0,2,0]

def count_presence_emolex_words(add_tokenized_list_of_list_here, emolex_file, emolex_tag):
	emolex_words = []
	with open(emolex_file, "r") as txtfile:
		for row in txtfile:
			row = row.lower().strip("\n").split("\t")
			if row[1] == emolex_tag:
				emolex_words.append(row[0])
			
	emolex_words = list(set(emolex_words))

	emolex_word_count = [] #This shows the number of emolex words for a selected emolex tag (e.g. negative) per tweet [0,3,0,2,......,5,1]
	for tweet in add_tokenized_list_of_list_here:
		emolex_tweet = []
		for w in emolex_words:
			for word in tweet: 
				if w == word.lower():
					emolex_tweet.append(w)	
		emolex_word_count.append(len(emolex_tweet)) 

	emolex_word_presence = [] #This shows the presence of emolex words for a selected emolex tag per tweet [0,1,0,1,......,1,1]
	for n in emolex_word_count:
		if n > 0:
			emolex_word_presence.append(1)
		else:
			emolex_word_presence.append(0)
	return emolex_word_count, emolex_word_presence

def tokenized_lowercased_input(add_X_with_space_no_hashtag_here):
	X_lowered_tokenized = []
	for x in add_X_with_space_no_hashtag_here:
		tweet_lowercased = [word.lower() for word in x]
		X_lowered_tokenized.append(tweet_lowercased)
	return X_lowered_tokenized

# Datasets and Resources
hurtlex_tr = read_tsvfile("hurtlex_TR.tsv")
offenseval_tr = read_tsvfile("offenseval-tr-training-v1.tsv")
offenseval_tr_testset = read_tsvfile("offenseval-tr-testset-v1.tsv")
karaliste_tr = read_txtfile("karaliste.txt")

def add_whitespace(data_list_of_list):
    #put a whitespace before and after punctuation 
    #Do not lowercase  

    selected_punctuations = '!\"#$%&()+,_-./:;<=>?[\]^`{|}~'
    text_with_space_before_punctuation = []

    for line in data_list_of_list:
        text_with_space_before_punctuation.append(["".join(line).translate(str.maketrans({key: " {0} ".format(key) for key in selected_punctuations}))])

    tokenized_text_with_space_before_punctuation = []
    for line in text_with_space_before_punctuation:
        # tokenized_text_with_space_before_punctuation.append("".join(line).strip().split())
        tokenized_text_with_space_before_punctuation.append("".join(line).strip().split())

    return tokenized_text_with_space_before_punctuation

def untokenized_preprocessed_input(X_with_space_no_hashtag_lowercased):
    X_untokenized_tweets = []
    for x in X_with_space_no_hashtag_lowercased:
        X_untokenized_tweets.append(" ".join(x))
    return X_untokenized_tweets

# INPUT, LABELS and FEATURES for TRAINING SET
#After writing a csv file (turkish_training_offense_features.csv) for the training set 
#including including input, features to be used and target values,
#uncomment the lines right after "X_with_space_no_hashtag_lowercased" to be used for 
#further preprocessing steps below
    
X,y = get_tweets_and_labels(offenseval_tr) 
X_normal_case = add_whitespace(remove_hashtag_symbols(X)) 
X_with_space_no_hashtag = add_whitespace_before_and_after_punctuation(remove_hashtag_symbols(X))
X_with_space_no_hashtag_lowercased = tokenized_lowercased_input(X_with_space_no_hashtag) 

"""
X_hashtag_presence = hashtag_presence(X)
X_user_mention_count = user_mention_count(add_whitespace_before_and_after_punctuation(X))
X_user_mention_presence = user_mention_presence(X_user_mention_count)

X_karaliste_words = karaliste_words(X_with_space_no_hashtag, karaliste_tr)
X_karaliste_word_count = karaliste_word_count(X_karaliste_words)
X_karaliste_word_presence = karaliste_word_presence(X_karaliste_word_count)

X_hurtlex_ps = count_hurtlex_words(X, "ps")
X_hurtlex_rci = count_hurtlex_words(X, "rci")
X_hurtlex_pa = count_hurtlex_words(X, "pa")
X_hurtlex_ddf = count_hurtlex_words(X, "ddf")
X_hurtlex_ddp = count_hurtlex_words(X, "ddp")
X_hurtlex_dmc = count_hurtlex_words(X, "dmc")
X_hurtlex_is = count_hurtlex_words(X, "is")
X_hurtlex_or = count_hurtlex_words(X, "or")
X_hurtlex_an = count_hurtlex_words(X, "an")
X_hurtlex_asm = count_hurtlex_words(X, "asm")
X_hurtlex_asf = count_hurtlex_words(X, "asf")
X_hurtlex_pr = count_hurtlex_words(X, "pr")
X_hurtlex_om = count_hurtlex_words(X,"om")
X_hurtlex_qas = count_hurtlex_words(X, "qas")
X_hurtlex_cds = count_hurtlex_words(X, "cds")
X_hurtlex_re = count_hurtlex_words(X, "re")
X_hurtlex_svp = count_hurtlex_words(X, "svp")

X_negative_emolex_count, X_negative_emolex_presence = count_presence_emolex_words(X_with_space_no_hashtag, "turkish_NRC_emolex.txt", "negative")
X_anger_emolex_count, X_anger_emolex_presence = count_presence_emolex_words(X_with_space_no_hashtag, "turkish_NRC_emolex.txt", "anger")

#Run this only once to write a csvfile for the training set including input,
#features to be used and target values
#output file: turkish_training_offense_features.csv
X_untokenized_lowercased_tweets = untokenized_preprocessed_input(X_with_space_no_hashtag_lowercased)
X_untokenized_normal_case_tweets = untokenized_preprocessed_input(X_normal_case)
    
import pandas as pd
df = pd.DataFrame({'untokenized_normal_case_tweets': X_untokenized_normal_case_tweets,
                       'tokenized_LOWERCASED_tweets': X_with_space_no_hashtag_lowercased,
					   'hashtag_presence': X_hashtag_presence,
					   'user_mention_count': X_user_mention_count,
					   'user_mention_presence': X_user_mention_presence,
					   'karaliste_word_count': X_karaliste_word_count,
					   'karaliste_word_presence': X_karaliste_word_presence,
					   'hurtlex_ps_count': X_hurtlex_ps,
					   'hurtlex_rci_count': X_hurtlex_rci,
					   'hurtlex_pa_count': X_hurtlex_pa,
					   'hurtlex_ddf_count': X_hurtlex_ddf,
					   'hurtlex_ddp_count': X_hurtlex_ddp,
					   'hurtlex_dmc_count': X_hurtlex_dmc,
					   'hurtlex_is_count': X_hurtlex_is,
					   'hurtlex_or_count': X_hurtlex_or,
					   'hurtlex_an_count': X_hurtlex_an,
					   'hurtlex_asm_count': X_hurtlex_asm,
					   'hurtlex_asf_count': X_hurtlex_asf,
					   'hurtlex_pr_count': X_hurtlex_pr,
					   'hurtlex_om_count': X_hurtlex_om,
					   'hurtlex_qas_count': X_hurtlex_qas,
					   'hurtlex_cds_count': X_hurtlex_cds,
					   'hurtlex_re_count': X_hurtlex_re,
					   'hurtlex_svp_count': X_hurtlex_svp,
					   'emolex_negative_count': X_negative_emolex_count,
					   'emolex_negative_presence': X_negative_emolex_presence,
					   'emolex_anger_count': X_anger_emolex_count,
					   'emolex_anger_presence': X_anger_emolex_presence,
                       'target_value': y})

df['tokenized_normal_case_tweets'] = df.apply(lambda row: nltk.word_tokenize(row['untokenized_normal_case_tweets']), axis=1)

df.to_csv("turkish_training_offense_features.csv", index=False, sep='\t', encoding='utf-8')
"""

# INPUT, FEATURES and TWEET IDS for TEST SET
#Afer writing a csv file (turkish_testset_offense_features.csv) for the testset 
#including features to be used and tweet ids, 
#uncomment the lines right after "Xtest_with_space_no_hashtag_lowercased"

Xtest_ids, Xtest_tweets = get_ids_and_tweets(offenseval_tr_testset)
Xtest_normal_case = add_whitespace(remove_hashtag_symbols(Xtest_tweets)) # this might be used for embeddings
Xtest_with_space_no_hashtag = add_whitespace_before_and_after_punctuation(remove_hashtag_symbols(Xtest_tweets)) # not a feature (used for count_presence_emolex_words), this can be used for word2vecs
Xtest_with_space_no_hashtag_lowercased = tokenized_lowercased_input(Xtest_with_space_no_hashtag) # used for TF-IDF

Xtest_hashtag_presence = hashtag_presence(Xtest_tweets)
Xtest_user_mention_count = user_mention_count(add_whitespace_before_and_after_punctuation(Xtest_tweets))
Xtest_user_mention_presence = user_mention_presence(Xtest_user_mention_count)

Xtest_karaliste_words = karaliste_words(Xtest_with_space_no_hashtag, karaliste_tr)
Xtest_karaliste_word_count = karaliste_word_count(Xtest_karaliste_words)
Xtest_karaliste_word_presence = karaliste_word_presence(Xtest_karaliste_word_count)

Xtest_hurtlex_ps = count_hurtlex_words(Xtest_tweets, "ps")
Xtest_hurtlex_rci = count_hurtlex_words(Xtest_tweets, "rci")
Xtest_hurtlex_pa = count_hurtlex_words(Xtest_tweets, "pa")
Xtest_hurtlex_ddf = count_hurtlex_words(Xtest_tweets, "ddf")
Xtest_hurtlex_ddp = count_hurtlex_words(Xtest_tweets, "ddp")
Xtest_hurtlex_dmc = count_hurtlex_words(Xtest_tweets, "dmc")
Xtest_hurtlex_is = count_hurtlex_words(Xtest_tweets, "is")
Xtest_hurtlex_or = count_hurtlex_words(Xtest_tweets, "or")
Xtest_hurtlex_an = count_hurtlex_words(Xtest_tweets, "an")
Xtest_hurtlex_asm = count_hurtlex_words(Xtest_tweets, "asm")
Xtest_hurtlex_asf = count_hurtlex_words(Xtest_tweets, "asf")
Xtest_hurtlex_pr = count_hurtlex_words(Xtest_tweets, "pr")
Xtest_hurtlex_om = count_hurtlex_words(Xtest_tweets,"om")
Xtest_hurtlex_qas = count_hurtlex_words(Xtest_tweets, "qas")
Xtest_hurtlex_cds = count_hurtlex_words(Xtest_tweets, "cds")
Xtest_hurtlex_re = count_hurtlex_words(Xtest_tweets, "re")
Xtest_hurtlex_svp = count_hurtlex_words(Xtest_tweets, "svp")

Xtest_negative_emolex_count, Xtest_negative_emolex_presence = count_presence_emolex_words(Xtest_with_space_no_hashtag, "turkish_NRC_emolex.txt", "negative")
Xtest_anger_emolex_count, Xtest_anger_emolex_presence = count_presence_emolex_words(Xtest_with_space_no_hashtag, "turkish_NRC_emolex.txt", "anger")

Xtest_untokenized_lowercased_tweets = untokenized_preprocessed_input(Xtest_with_space_no_hashtag_lowercased)
Xtest_untokenized_normal_case_tweets = untokenized_preprocessed_input(Xtest_normal_case)
    

#write a csv file for testset features and tweet ids
#output file: turkish_testset_offense_features.csv
df = pd.DataFrame({'user_mention_presence': Xtest_user_mention_presence,
					   'karaliste_word_count': Xtest_karaliste_word_count,
					   'karaliste_word_presence': Xtest_karaliste_word_presence,
					   'hurtlex_ddp_count': Xtest_hurtlex_ddp,
					   'hurtlex_dmc_count': Xtest_hurtlex_dmc,
					   'hurtlex_or_count': Xtest_hurtlex_or,
					   'hurtlex_asm_count': Xtest_hurtlex_asm,
					   'hurtlex_asf_count': Xtest_hurtlex_asf,
					   'hurtlex_om_count': Xtest_hurtlex_om,
					   'hurtlex_cds_count': Xtest_hurtlex_cds,
					   'emolex_negative_count': Xtest_negative_emolex_count,
                       'tweet_ids': Xtest_ids})

df.to_csv("turkish_testset_offense_features.csv", index=False, sep='\t', encoding='utf-8')

# FURTHER PREPROCESSING STEPS TO BE APPLIED ON X_with_space_no_hashtag_lowercased (for training set)
#                                              Xtest_with_space_no_hashtag_lowercased (for test set)


def remove_mention_tags(text_in_list):
    # @user will be user
    text_without_mention_tags = [] 
    for line in text_in_list:
        line_string = " ".join(line)
        rexp_string = r'@user'
        tags_created_rexp = re.compile(rexp_string)
        text_cleaned = tags_created_rexp.sub("user",line_string)
        text_without_mention_tags.append(text_cleaned)
    return text_without_mention_tags

def remove_punctuation(data_in_list):
    #remove punctuations
    X_without_punctuation = []
    for line in data_in_list:
        line = line.translate(str.maketrans('', '', string.punctuation))
        X_without_punctuation.append(line.split())  
    return X_without_punctuation

def replace_numbers(list_of_list):
    #tag numbers with "number" tag
    text_with_number_label = []
    for line in list_of_list:
        line_string = " ".join(line)
        rexp_string = r'([0-9]+)'
        compiled_numbers = re.compile(rexp_string)
        text_modified = compiled_numbers.sub("number",line_string)
        text_with_number_label.append(text_modified.split())
    return text_with_number_label

def tokenized_tweets():
    return X_with_space_no_hashtag_lowercased

def tokenized_tweets_testset():
    return Xtest_with_space_no_hashtag_lowercased

def preprocessed_training_for_neural_network():
    # This is the preprocessed input for the training set
    return replace_numbers(remove_punctuation(remove_mention_tags(tokenized_tweets())))
    
def preprocessed_testset_for_neural_network():
    # This is the preprocessed input for the test set
    return replace_numbers(remove_punctuation(remove_mention_tags(tokenized_tweets_testset())))

def training_dataset_features_labels():
    dataset = pd.read_csv('turkish_training_offense_features.csv',sep='\t')
    X_features = dataset.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]].values.astype('int')
    y = dataset.iloc[:, 28].values
    
    return dataset, X_features, y

def testset_dataset_features_labels():
    test_dataset = pd.read_csv('turkish_testset_offense_features.csv',sep='\t')
    Xtest_features = test_dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,11]].values.astype('int')
    Xtest_ids = test_dataset.iloc[:, 10].values
    
    return test_dataset, Xtest_features, Xtest_ids

def to_categorical_labels():
    _,_,y = training_dataset_features_labels()
    y_labels = [] # LABEL BINARIZER ==> 0 FOR NOT, 1 FOR OFF
    for l in y:
        labels = []
        if l == "NOT":
            labels.append(0)
        else:
            labels.append(1)
        y_labels += labels
    # to_categorical: Converts a class vector (integers) to binary class matrix.
    labels = to_categorical(np.asarray(y_labels))
    return y_labels, labels

def X_features_dense():
    dataset, _, _ = training_dataset_features_labels()
    # convert numpy array to sparse csr matrix
    
    X_4 = sparse.csr_matrix(dataset.iloc[:,[4]].values.astype('int'))
    X_5 = sparse.csr_matrix(dataset.iloc[:,[5]].values.astype('int'))
    X_6 = sparse.csr_matrix(dataset.iloc[:,[6]].values.astype('int'))
    X_11 = sparse.csr_matrix(dataset.iloc[:,[11]].values.astype('int'))
    X_12 = sparse.csr_matrix(dataset.iloc[:,[12]].values.astype('int'))
    X_14 = sparse.csr_matrix(dataset.iloc[:,[14]].values.astype('int'))
    X_16 = sparse.csr_matrix(dataset.iloc[:,[16]].values.astype('int'))
    X_17 = sparse.csr_matrix(dataset.iloc[:,[17]].values.astype('int'))
    X_19 = sparse.csr_matrix(dataset.iloc[:,[19]].values.astype('int'))
    X_21 = sparse.csr_matrix(dataset.iloc[:,[21]].values.astype('int'))
    X_24 = sparse.csr_matrix(dataset.iloc[:,[24]].values.astype('int'))
    
    # The features above are selected features which will be converted into dense format
    X_4_dense = X_4.toarray()
    X_5_dense = X_5.toarray()
    X_6_dense = X_6.toarray()
    X_11_dense = X_11.toarray()
    X_12_dense = X_12.toarray()
    X_14_dense = X_14.toarray()
    X_16_dense = X_16.toarray()
    X_17_dense = X_17.toarray()
    X_19_dense = X_19.toarray()
    X_21_dense = X_21.toarray()
    X_24_dense = X_24.toarray()
    
    return X_4_dense, X_5_dense, X_6_dense, X_11_dense, X_12_dense, X_14_dense, X_16_dense, X_17_dense, X_19_dense, X_21_dense, X_24_dense


def Xtest_features_dense():
    dataset, _, _ = testset_dataset_features_labels()
    # convert numpy array to sparse csr matrix
    X_0 = sparse.csr_matrix(dataset.iloc[:,[0]].values.astype('int'))
    X_1 = sparse.csr_matrix(dataset.iloc[:,[1]].values.astype('int'))
    X_2 = sparse.csr_matrix(dataset.iloc[:,[2]].values.astype('int'))
    X_3 = sparse.csr_matrix(dataset.iloc[:,[3]].values.astype('int'))
    X_4 = sparse.csr_matrix(dataset.iloc[:,[4]].values.astype('int'))
    X_5 = sparse.csr_matrix(dataset.iloc[:,[5]].values.astype('int'))
    X_6 = sparse.csr_matrix(dataset.iloc[:,[6]].values.astype('int'))
    X_7 = sparse.csr_matrix(dataset.iloc[:,[7]].values.astype('int'))
    X_8 = sparse.csr_matrix(dataset.iloc[:,[8]].values.astype('int'))
    X_9 = sparse.csr_matrix(dataset.iloc[:,[9]].values.astype('int'))
    X_11 = sparse.csr_matrix(dataset.iloc[:,[11]].values.astype('int'))
    
    # The features above are selected features which will be converted into dense format
    X_0_dense = X_0.toarray()
    X_1_dense = X_1.toarray()
    X_2_dense = X_2.toarray()
    X_3_dense = X_3.toarray()
    X_4_dense = X_4.toarray()
    X_5_dense = X_5.toarray()
    X_6_dense = X_6.toarray()
    X_7_dense = X_7.toarray()
    X_8_dense = X_8.toarray()
    X_9_dense = X_9.toarray()
    X_11_dense = X_11.toarray()
    
    return X_0_dense, X_1_dense, X_2_dense, X_3_dense, X_4_dense, X_5_dense, X_6_dense, X_7_dense, X_8_dense, X_9_dense, X_11_dense
"""
______                                                  _             _   _ _   _ _
| ___ \                                                | |           | | | | | (_) |
| |_/ /___  ___ ___  _ __ ___  _ __ ___   ___ _ __   __| | ___ _ __  | | | | |_ _| |___
|    // _ \/ __/ _ \| '_ ` _ \| '_ ` _ \ / _ \ '_ \ / _` |/ _ \ '__| | | | | __| | / __|
| |\ \  __/ (_| (_) | | | | | | | | | | |  __/ | | | (_| |  __/ |    | |_| | |_| | \__ \
\_| \_\___|\___\___/|_| |_| |_|_| |_| |_|\___|_| |_|\__,_|\___|_|     \___/ \__|_|_|___/


A collection of helper functions intended for use with the LightFM hybrid recommender model.

Version 0.0.1
"""

# import modules
import os
import zipfile
import csv
import pandas as pd
import requests
import json
import numpy as np
import nltk
import string
import re
import random
import operator
import time
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import islice, chain

from collections import Counter

from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank
from lightfm import LightFM
from lightfm.data import Dataset

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold

from scipy import sparse

from ast import literal_eval


def get_similar_tags(model, tag_id):
    tag_embeddings = (model.item_embeddings.T
                      / np.linalg.norm(model.item_embeddings, axis=1)).T

    query_embedding = tag_embeddings[tag_id]
    similarity = np.dot(tag_embeddings, query_embedding)
    most_similar = np.argsort(-similarity)[1:4]

    return most_similar


def similar_items(item_id, item_features, model, N=10):
    """
    Function to determine most similar items based on learned item embeddings within recommender model
    :param item_id: Item id for the item you want to compare to other items. This is the internal model id which you need to extract first
    :param item_features: Sparse matrix of time features used in the model
    :param model: The trained LightFM recommender model
    :param N: Number of items to return
    :return: List of tuples of N most similar items by internal item id and the similarity score
    """
    (item_biased, item_representations) = model.get_item_representations(features=item_features)
    # Cosine similarity
    scores = item_representations.dot(item_representations[item_id])
    item_norms = np.linalg.norm(item_representations, axis=1)
    item_norms[item_norms == 0] = 1e-10
    scores /= item_norms
    best = np.argpartition(scores, -N)[-N:]

    return sorted(zip(best, scores[best] / item_norms[item_id]), key=lambda x: -x[1])


def convertlist(longlist):
    """
    Function to convert list of lists to a single numpy array. Works on an iterable to be more efficient than working directly with lists
    :param longlist: List of lists
    :return: Numpy array with elements from the lists
    """
    tmp = list(chain.from_iterable(longlist))
    return np.array(tmp).reshape((len(longlist), len(longlist[0])))


def prepare_interactions_dataframe(df, user_id_col, item_id_col, weight_col=None, interaction_weights=False, item_identity_features=False, user_identity_features=False):
    """
    Function to prepare input dataframe containing user-item interactions for use within LightFM model
    :param df: Dataframe with user-item interactions
    :param user_id_col: Name of column containing the user ids
    :param item_id_col: Name of column containing the item ids
    :param weight_col: Name of column containing user-item weights. For example, the rating that a user gives to an item
    :param interaction_weights: Boolean indicating whether or not to use explicit users weights.  There are instances where this may not be desirable
    :param item_identity_features: Boolean to indicate whether or not to append an identify matrix to the item features matrix on top of existing item features. True helps with cold start problems
    :param user_identity_features: Boolean to indicate whether or not to append an identify matrix to the user features matrix on top of existing user features. True helps with cold start problems
    :return: LightFM dataset object, interactions matrix (scipy sparse coo matrix), matrix of weights (scipy sparse coo matrix) containing interaction weights
             (identity matrix if no interaction weights are desired)
    """

    # Initialize the dataset. Identity features are off by default here
    dataset = Dataset(item_identity_features=item_identity_features, user_identity_features=user_identity_features)

    # fit the primary interactions
    dataset.fit(users=df[user_id_col], items=df[item_id_col])

    # Characterize the size of the interactions
    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}'.format(num_users, num_items))

    # Build the interactions and weights matrices. Weights will be optional
    if interaction_weights:
        (interactions, weights) = dataset.build_interactions(list(zip(df[user_id_col],
                                                                      df[item_id_col],
                                                                      df[weight_col].astype(int))))
    else:
        (interactions, weights) = dataset.build_interactions(list(zip(df[user_id_col],
                                                                      df[item_id_col])))

    return dataset, interactions, weights


def prepare_item_features(dataset, item_features_df, item_id_col, feature_col, features_list):
    """
    Function to prepare item features for use with LightFM model from a dataframe
    :param dataset: LightFM dataset object returned by prepare_interactions_dataframe function
    :param item_features_df: Dataframe containing the item features. One item per row
    :param item_id_col: Name of the column containing the item ids
    :param feature_col: Name of the column containing the item features. Expected to be a list of features, or a list containing a dictionary of the form {feature name: weight}
           Otherwise, the default will be a hot encoding based on provided feature names
    :param features_list: List of unique feature names
    :return: Scipy sparse matrix of items and item features. Entries will either be a hot encoding if a list of feature names per item is passed,
             or explicit weights if a dictionary of feature names to weights per item is provided
    """

    # TODO: Add support for arbitrary feature weights. This should work already as long as the feature_col is formatted like {feature name: weight}
    # Build (item id, list of features for that item) tuple.
    feature_tuple = tuple(zip(item_features_df[item_id_col], item_features_df[feature_col]))

    # perform a partial fit of the item ids and item features
    dataset.fit_partial(items=item_features_df[item_id_col],
                        item_features=features_list)

    # Add item features to the dataset already initialized.  Must run prepare_interactions_dataframe function first.
    item_features = dataset.build_item_features(feature_tuple)

    return item_features


def prepare_user_features(dataset, user_features_df, user_id_col, feature_col, features_list):
    """
    Function to prepare user features for use with LightFM model
    :param dataset: LightFM dataset object returned by prepare_interactions_dataframe function
    :param user_features_df: Dataframe containing the user features. One user per row
    :param user_id_col: Name of the column containing the user ids
    :param feature_col: Name of the column containing the user features. Expected to be a list of features, or a list containing a dictionary of the form {feature name: weight}.
           Otherwise, the default will be a hot encoding based on provided feature names
    :param features_list: List of unique feature names
    :return: Scipy sparse matrix of users and user features. Entries will either be a hot encoding if a list of feature names per user is passed,
             or explicit weights if a dictionary of feature names to weights per user is provided
    """

    # TODO: Add support for arbitrary feature weights. This should work already as long as the feature_col is formatted like {feature name: weight}
    # Build (user id, list of features for that item) tuple.
    feature_tuple = tuple(zip(user_features_df[user_id_col], user_features_df[feature_col]))

    # perform a partial fit of the item ids and item features
    dataset.fit_partial(users=user_features_df[user_id_col],
                        user_features=features_list)

    # Add user features to the dataset already initialized.  Must run prepare_interactions_dataframe function first.
    user_features = dataset.build_user_features(feature_tuple)

    return user_features


def make_feature_lists(df, feature_list_cols):
    """
    Function to extract unique features
    :param df:
    :param feature_list_cols:
    :return:
    """
    # TODO: decide if we actually need this here.  Might be something that needs to be custom to the problem at hand
    # Might move this to another module as this is a bit custom to the problem
    # Will need to handle numeric, categorial, and text data
    pass


def build_item_column(df, feature_cols):
    """
    Function to build the list of features for each item or user
    :param df:
    :param feature_cols:
    :return:
    """
    # Might put this in another module as this is a bit custom to the problem
    # Will need to return a list if just feature names, or a list with a dictionary with feature name: weight if custom weights are required.
    pass


def prepare_test_interactions(dataset, df, user_id_col, item_id_col, weight_col=None, interaction_weights=False):
    """
    Function to prepare a test set for model evaluation. Should only be run with users and items that are already in the training set, but previously unseen interactions.
    :param dataset: LightFM dataset object returned by prepare_interactions_dataframe function
    :param df: Dataframe with user-item interactions
    :param user_id_col: Name of column containing the user ids
    :param item_id_col: Name of column containing the item ids
    :param weight_col: Name of column containing user-item weights. For example, the rating that a user gives to an item
    :param interaction_weights: Boolean indicating whether or not to use explicit users weights.
    :return: interactions matrix (scipy sparse coo matrix), matrix of weights (scipy sparse coo matrix) containing interaction weights (identity matrix if no interaction weights are desired)
    """

    # TODO: Check if we actually need to return in the interaction weights here.  Not really all that relevant for testing purposes.
    # Build the interactions and weights matrices. Weights will be optional
    if interaction_weights:
        (interactions, weights) = dataset.build_interactions(list(zip(df[user_id_col],
                                                                      df[item_id_col],
                                                                      df[weight_col].astype(int))))
    else:
        (interactions, weights) = dataset.build_interactions(list(zip(df[user_id_col],
                                                                      df[item_id_col])))

    return interactions, weights


def make_cold_start_data(df, user_id_col, item_id_col, rating_col, shape, types):
    """
    A function to convert cold start data to the correct format for usage with LightFM model
    :param df: Dataframe with user-item interactions. Can contain new users not previously seen by model
    :param user_id_col: Name of column containing the user ids
    :param item_id_col: Name of column containing the item ids
    :param rating_col: Name of column containing user-item ratings
    :param shape: Shape of interaction matrix to return.  Needs to be the same shape as the interactions matrix returned by prepare_interactions_dataframe function
    :param types: Types for the entries in the columns.  Needs to be the same as the data types in the interactions matrix returned by prepare_interactions_dataframe function
    :return: Interaction matrix from cold start user-item dataframe (scipy sparse coo matrix)
    """

    # TODO: check is this is really needed.  We could probably just use the built in train-test split function from LightFM to achieve this.
    df_pivot = df.pivot(index=user_id_col, columns=item_id_col, values=rating_col)
    df_pivot.fillna(0, inplace=True)
    df_pivot = df_pivot.astype(np.int32)

    interactions = sparse.csr_matrix(df_pivot.values)
    interactions = interactions.tocoo()
    uids, iids, data = (interactions.row, interactions.col, interactions.data)
    interactions = sparse.coo_matrix((data, (uids, iids)), shape=shape, dtype=types)

    return interactions


def prepare_item_features_from_dict(dataset, item_features_dict, features_list):
    """
    Function to prepare item features for use with LightFM model from a dictionary
    :param dataset: LightFM dataset object returned by prepare_interactions_dataframe function
    :param item_features_dict: Dictionary with keys being the item ids and values the features.  Features should be either a list of feature names (this defaults the model to hot encoding) or a
                               dictionary {feature name: feature weight}.
    :param features_list: List of individual feature names
    :return: Scipy sparse matrix of items and item features. Entries will either be a hot encoding if a list of feature names per item is passed,
             or explicit weights if a dictionary of feature names to weights per item is provided
    """

    # Build (item id, list of features for that item) tuple.
    feature_tuple = tuple(zip(item_features_dict.keys(), item_features_dict.values()))

    # perform a partial fit of the item ids and item features
    dataset.fit_partial(items=item_features_dict.keys(),
                        item_features=features_list)

    # Add item features to the dataset already initialized.  Must run prepare_interactions_dataframe function first.
    try:
        item_features = dataset.build_item_features(feature_tuple)
    except ValueError as e:
        if str(e) != 'Cannot normalize feature matrix: some rows have zero norm.':
            raise
        else:
            print('Cannot normalize feature matrix: some rows have zero norm.')
            print('Defaulting to non-normalized features, but best to check feature matrix.')
            item_features = dataset.build_item_features(feature_tuple, normalize=False)

    return item_features


def prepare_user_features_from_dict(dataset, user_features_dict, features_list):
    """
    Function to prepare user features for use with LightFM model from a dictionary
    :param dataset: LightFM dataset object returned by prepare_interactions_dataframe function
    :param user_features_dict: Dictionary with keys being the user ids and values the features.  Features should be either a list of feature names (this defaults the model to hot encoding) or a
                               dictionary {feature name: feature weight}.
    :param features_list: List of individual feature names
    :return: Scipy sparse matrix of users and user features. Entries will either be a hot encoding if a list of feature names per user is passed,
             or explicit weights if a dictionary of feature names to weights per user is provided
    """

    # Build (item id, list of features for that item) tuple.
    feature_tuple = tuple(zip(user_features_dict.keys(), user_features_dict.values()))

    # perform a partial fit of the item ids and item features
    dataset.fit_partial(users=user_features_dict.keys(),
                        user_features=features_list)

    # Add user features to the dataset already initialized.  Must run prepare_interactions_dataframe function first.
    try:
        user_features = dataset.build_user_features(feature_tuple)
    except ValueError as e:
        if str(e) != 'Cannot normalize feature matrix: some rows have zero norm.':
            raise
        else:
            print('Cannot normalize feature matrix: some rows have zero norm.')
            print('Defaulting to non-normalized features, but best to check feature matrix.')
            user_features = dataset.build_user_features(feature_tuple, normalize=False)

    return user_features
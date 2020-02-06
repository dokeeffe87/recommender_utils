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
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from itertools import chain

from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank
from lightfm.data import Dataset

from scipy import sparse, stats


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
    :param item_identity_features: Boolean to indicate whether or not to append an identify matrix to the item features matrix on top of existing item features.
                                   True lets you fall back on collaborative filtering
    :param user_identity_features: Boolean to indicate whether or not to append an identify matrix to the user features matrix on top of existing user features.
                                   True lets you fall back on collaborative filtering
    :return: LightFM dataset object, interactions matrix (scipy sparse coo matrix), matrix of weights (scipy sparse coo matrix) containing interaction weights
             (identity matrix if no interaction weights are desired)
    """
    # TODO: Should I set the default for identity features as True?
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
    # Will need to handle numeric, categorical, and text data
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

    # TODO: Check if we actually need to return the interaction weights here.  Not really all that relevant for testing purposes.
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

    # TODO: check if this is really needed.  We could probably just use the built in train-test split function from LightFM to achieve this.
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


def compute_eval_metrics(model, train_interactions, test_interactions, item_features=None, user_features=None, tuple_of_k=(10, 20, 50), compute_on_train=False, preserve_rows=False, check_intersections=False, num_threads=1):
    """
    Function to compute standard evaluation metrics on given test set and model
    :param model: The model to test
    :param train_interactions: The train set of user/item interactions (sparse matrix).  This is used to remove training examples from the test set score calculations which could bias results
    :param test_interactions: The test set of user/item interactions (sparse matrix)
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :param tuple_of_k: A tuple containing the values of k you want to consider for the precision at k and recall at k metrics
    :param compute_on_train: Set true if you want to see the metrics computed on the training set as well
    :param preserve_rows: Preserve rows parameter in the LightFM provided metrics: When False (default), the number of rows in the output will be equal to the number of users with interactions in the
                          evaluation set. When True, the number of rows in the output will be equal to the number of users
    :param check_intersections: Check for intersecting train and test interactions
    :param num_threads: The number of threads to use
    :return: dictionary with the metrics as keys and the list of results as values
    """

    # Initialize a dictionary to store results
    all_results_dict = {}

    # Compute the test_auc:
    print('Computing test set AUCs')
    test_auc_list = auc_score(model=model,
                              train_interactions=train_interactions,
                              test_interactions=test_interactions,
                              item_features=item_features,
                              user_features=user_features,
                              preserve_rows=preserve_rows,
                              check_intersections=check_intersections,
                              num_threads=num_threads)
    all_results_dict['test_auc_list'] = test_auc_list

    # Compute test recall at all input k values:
    for k in tuple_of_k:
        print('Computing test set recalls at {0}'.format(k))
        test_recall_list = recall_at_k(model=model,
                                       train_interactions=train_interactions,
                                       test_interactions=test_interactions,
                                       item_features=item_features,
                                       user_features=user_features,
                                       preserve_rows=preserve_rows,
                                       check_intersections=check_intersections,
                                       num_threads=num_threads,
                                       k=k)
        all_results_dict['test_recall_at_{0}_list'.format(k)] = test_recall_list

    # Compute test precision at all input k values:
    for k in tuple_of_k:
        print('Computing test set precisions at {0}'.format(k))
        test_precision_list = precision_at_k(model=model,
                                             train_interactions=train_interactions,
                                             test_interactions=test_interactions,
                                             item_features=item_features,
                                             user_features=user_features,
                                             preserve_rows=preserve_rows,
                                             check_intersections=check_intersections,
                                             num_threads=num_threads,
                                             k=k)
        all_results_dict['test_precision_at_{0}_list'.format(k)] = test_precision_list

    # Compute test reciprocal rank
    print('Computing test set reciprocal rank')
    test_reciprocal_rank_list = reciprocal_rank(model=model,
                                                train_interactions=train_interactions,
                                                test_interactions=test_interactions,
                                                item_features=item_features,
                                                user_features=user_features,
                                                preserve_rows=preserve_rows,
                                                check_intersections=check_intersections,
                                                num_threads=num_threads)
    all_results_dict['test_reciprocal_rank_list'] = test_reciprocal_rank_list

    if compute_on_train:
        print('Computing metrics on training set as well')

        # Compute the train_auc:
        print('Computing train set AUCs')
        train_auc_list = auc_score(model=model,
                                   test_interactions=train_interactions,
                                   item_features=item_features,
                                   user_features=user_features,
                                   preserve_rows=preserve_rows,
                                   num_threads=num_threads)
        all_results_dict['train_auc_list'] = train_auc_list

        # Compute train recall at all input k values:
        for k in tuple_of_k:
            print('Computing train set recalls at {0}'.format(k))
            train_recall_list = recall_at_k(model=model,
                                            test_interactions=train_interactions,
                                            item_features=item_features,
                                            user_features=user_features,
                                            preserve_rows=preserve_rows,
                                            num_threads=num_threads,
                                            k=k)
            all_results_dict['train_recall_at_{0}_list'.format(k)] = train_recall_list

        # Compute train precision at all input k values:
        for k in tuple_of_k:
            print('Computing train set precisions at {0}'.format(k))
            train_precision_list = precision_at_k(model=model,
                                                  test_interactions=train_interactions,
                                                  item_features=item_features,
                                                  user_features=user_features,
                                                  preserve_rows=preserve_rows,
                                                  num_threads=num_threads,
                                                  k=k)
            all_results_dict['train_precision_at_{0}_list'.format(k)] = train_precision_list

        # Compute train reciprocal rank
        print('Computing train set reciprocal rank')
        train_reciprocal_rank_list = reciprocal_rank(model=model,
                                                     test_interactions=train_interactions,
                                                     item_features=item_features,
                                                     user_features=user_features,
                                                     preserve_rows=preserve_rows,
                                                     num_threads=num_threads)
        all_results_dict['train_reciprocal_rank_list'] = train_reciprocal_rank_list

    return all_results_dict


def compute_eval_metric_summaries(eval_dict):
    """
    Given the computed evaluation metrics for each user in a set, this will produce a summary dataframe of descriptive performance statistics
    :param eval_dict: The evaluation dictionary returned by the compute_eval_metric function
    :return: dataframe with the summary stats of the evaluation metrics
    """

    # Initialize a dictionary to store final results
    storage_dict = {}

    for metric_, value_list in eval_dict.items():
        # Compute
        nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(value_list)
        median = np.median(value_list)
        percentile_25 = np.percentile(value_list, q=25)
        percentile_75 = np.percentile(value_list, q=50)

        storage_dict[metric_[:-5]] = {'number_observations': nobs,
                                      'min': minmax[0],
                                      'max': minmax[1],
                                      'mean': mean,
                                      'variance': variance,
                                      'skewness': skewness,
                                      'kurtosis': kurtosis,
                                      'median': median,
                                      '25th percentile': percentile_25,
                                      '75th percentile': percentile_75}

    results_df = pd.DataFrame.from_dict(storage_dict, orient='index')
    results_cols = results_df.columns.tolist()
    results_df.reset_index(inplace=True)
    results_df.columns = ['metric'] + results_cols

    return results_df


def plot_metric_dist(eval_dict, metric, data_split='test', figsize=(20, 10), title=None, kde=False, save_name=None, axis_fontsize=10, title_fontsize=10):
    """
    Function to plot the distribution of given metrics
    :param eval_dict: The evaluation metric dictionary returned by the compute_eval_metric function
    :param metric: The metric you want to see the distribution for
    :param data_split: If the provided values are on the test set or the train set
    :param figsize: Tuple which controls the vertical and horizontal figure size
    :param title: Optional string for the chart title
    :param kde: Optional boolean. True will fit a kernel density estimate to the distribution
    :param save_name: Path and file name you want to use to save the output charts. Should ideally be a pdf file.  Optional
    :param axis_fontsize: Font size for the axes labels
    :param title_fontsize: Font size for the title
    :return: Nothing. Just makes the desired plot and saves it if directed to
    """

    metric = metric.lower()
    data_split = data_split.lower()

    metric_key = data_split + '_' + metric + '_list'

    if metric_key not in eval_dict.keys():
        raise ValueError('Invalid metric and/or data_split.')

    fig, ax = plt.subplots(figsize=figsize)
    if title is not None:
        sns.distplot(eval_dict[metric_key], kde=kde, ax=ax)
        ax.set_title(title, fontsize=title_fontsize)
    else:
        sns.distplot(eval_dict[metric_key], kde=kde, ax=ax)

    # Format x-axis name
    metric_name_list = metric_key[:-5].split('_')
    if metric_name_list[0] == 'test':
        metric_name_list = ['Test'] + metric_name_list[1::]
    else:
        metric_name_list = ['Train'] + metric_name_list[1::]

    xlabel = " ".join(metric_name_list)

    ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    ax.set_ylabel('count', fontsize=axis_fontsize)

    if save_name is not None:
        plt.savefig(save_name)

    plt.show()


def predict_on_users(model, dataset, list_of_user_ids, list_of_item_ids, item_features=None, user_features=None):
    """
    Make a recommendation prediction on a set of users and a set of items
    :param model: The recommender model to use to make the predictions
    :param dataset: The dataset object created before fitting the LightFM model
    :param list_of_user_ids: A list of user ids from your dataset that you want to predict on.  The ids must exist in the data mapping, so you may need to run a partial_fit on these user ids
    :param list_of_item_ids: A list of items ids from your dataset that you want to predict on.  The ids must exist in the data mapping, so you may need to run a partial_fit on these item ids
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :return: A dataframe with the predicted score for each user-item pair
    """

    # Convert the list of user ids and item ids to the internal indices used by the model
    internal_user_id_list = np.array([dataset.mapping()[0][id_] for id_ in list_of_user_ids])
    internal_item_id_list = np.array([dataset.mapping()[2][id_] for id_ in list_of_item_ids])

    # Invert the dictionaries for reverse translation
    internal_id_to_user_dict = {val: key for key, val in dataset.mapping()[0].items()}
    internal_id_to_item_dict = {val: key for key, val in dataset.mapping()[2].items()}

    pred_dict = {}

    for id_ in internal_user_id_list:
        preds_ = model.predict(user_ids=np.array([id_ for x in internal_item_id_list]), item_ids=internal_item_id_list, item_features=item_features, user_features=user_features)
        pred_dict[id_] = {}
        for i in range(len(preds_)):
            pred_dict[id_][internal_item_id_list[i]] = preds_[i]

    # Convert to a dataframe
    preds_list = [[key, val_0, val_1] for key, value in pred_dict.items() for val_0, val_1 in value.items()]
    preds_df = pd.DataFrame(preds_list)
    preds_df.columns = ['user_id_internal', 'item_id_internal', 'score']

    # Map back to the external ids
    preds_df['user_id'] = preds_df['user_id_internal'].map(internal_id_to_user_dict)
    preds_df['item_id'] = preds_df['item_id_internal'].map(internal_id_to_item_dict)

    # Drop the internal ids as we don't need them anymore
    preds_df.drop(['user_id_internal', 'item_id_internal'], axis=1, inplace=True)

    return preds_df


def format_recommendations(predictions_df, item_features_df=None, user_features_df=None, item_features_col=None, user_features_col=None, item_id_col=None, user_id_col=None):
    """
    Formats the recommendation results in the dataframe produced by the predict_on_users function
    :param predictions_df: The dataframe produced by the predict_on_users function
    :param item_features_df: The original dataframe that contains the item ids and the features that went into the item_features sparse matrix
    :param user_features_df: The original dataframe that contains the user ids and the features that went into the user_features sparse matrix
    :param item_features_col: The name of the column in the item_features_df you want to us to describe the items being considered (i.e. study title, book title, etc).
    :param user_features_col: The name of the column in the user_features_df you want to us to describe the users being considered (i.e. name, twitter handle, etc).
    :param item_id_col: Column name in the item_features_df which contains the original item ids
    :param user_id_col: Column name in the user_features_df which contains the original user ids
    :return: Nothing. Just prints the results to the screen
    """
    if isinstance(item_features_df, pd.core.frame.DataFrame):
        if not item_features_col:
            raise ValueError('Please supply a column from the item features dataframe to find the desired item descriptions')
        if not item_id_col:
            raise ValueError('Please supply the item id column name in the item features dataframe')
    if isinstance(user_features_df, pd.core.frame.DataFrame):
        if not user_features_col:
            raise ValueError('Please supply a column from the user features dataframe to find the desired user descriptions')
        if not user_id_col:
            raise ValueError('Please supply the user id column name in the user features dataframe')

    grped_df = predictions_df.groupby('user_id')

    for id_, grp in grped_df:
        df_ = grp.sort_values('score', ascending=False).reset_index()
        print('Recommendations for input ID: {0}'.format(id_))
        if user_features_col:
            print('User description: {0}'.format(user_features_df[user_features_col].loc[(user_features_col[user_id_col] == id_)].values[0]))
        for i in range(df_.shape[0]):
            item = df_['item_id'].iloc[i]
            # We'll leave the score as is for now, but we may want to normalize the scale at some point
            score = df_['score'].iloc[i]
            print('     Recommended item id: {0}'.format(item))
            if item_features_col:
                print('          Item description: {0}'.format(item_features_df[item_features_col].loc[(item_features_df[item_id_col] == item)].values[0]))
            print('          Score: {0}'.format(score))
            print('---------------------------------------------------')
        print('\n')


def save_data_objects(filename, dataset=None, matrix=None):
    """
    Function to save data objects used for the model
    :param filename: Directory and name of the file you want to save data as. Dataset objects should be .pickle files. Matrix objects should be .npz files
    :param dataset: Dataset object used by LightFM. Should be saved as a .pickle file
    :param matrix: A scipy sparse matrix containing data.  Should be a .npz file
    :return: Nothing. Saves the input file at the specified directory + file name
    """
    # TODO: Add support for multiple saves at once.
    if dataset is not None:
        with open('filename', 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    if matrix is not None:
        sparse.save_npz(filename, matrix)


def load_data_objects(filename, is_dataset=False, is_matrix=False):
    """
    Function to load both LightFM dataset objects and data matrices
    :param filename: Path and name of the file you want to load
    :param is_dataset: Set True if the file is a dataset object
    :param is_matrix: Set true if the file is a data matrix
    :return: Either LightFM dataset object or a scipy sparse matrix
    """
    # TODO: Add support for multiple loads at once
    if is_dataset:
        dataset = pickle.load(open(filename, 'rb', -1))

        return dataset

    if is_matrix:
        matrix = sparse.load_npz(filename)

        return matrix


def make_long_tail_plot(df, item_id_column, save_name, interaction_type, take_top_n=None, percentage=None, x_labels=True):
    """
    Makes a long tail plot of the interactions data. Inspired by: https://github.com/statisticianinstilettos/recmetrics
    :param df:
    :param item_id_column:
    :param save_name:
    :param interaction_type:
    :param take_top_n:
    :param percentage:
    :param x_labels:
    :return: None. Saves a plot instead. Doesn't display the plot since this could cause rendering issues with large datasets.
    """
    # TODO: Make this more scalable. It struggles with 100,000+ data points.
    volume_df = pd.DataFrame(df[item_id_column].value_counts())
    volume_df.reset_index(inplace=True)
    volume_df.columns = [item_id_column, "volume"]
    volume_df[item_id_column] = volume_df[item_id_column].astype(str)
    volume_df['cumulative_volume'] = volume_df['volume'].cumsum()
    volume_df['percent_of_total_volume'] = volume_df['cumulative_volume'] / volume_df['volume'].sum()
    volume_df.sort_values(by='volume', ascending=False)

    if take_top_n is not None:
        volume_df = volume_df.head(take_top_n)
    else:
        print("Using all interaction data.  This could result in long run times for large datasets.")
        print('Consider using the take_top_n argument to limit to the top n interactions if you have more than 10,000 in your dataset.')

    # line plot of cumulative volume
    x = range(0, len(volume_df))
    f = plt.figure(figsize=(20, 10))
    ax = sns.lineplot(x, y="volume", data=volume_df, color="black")
    plt.xticks(x)

    # set labels
    ax.set_title('Long Tail Plot')
    ax.set_ylabel('# of ' + interaction_type)
    ax.set_xlabel(item_id_column)
    ax.tick_params(axis='x', labelrotation=90)

    if percentage is not None:
        # plot vertical line at the tail location
        head = volume_df[volume_df.percent_of_total_volume <= percentage]
        tail = volume_df[volume_df.percent_of_total_volume > percentage]
        items_in_head = len(head)
        items_in_tail = len(tail)
        plt.axvline(x=items_in_head, color="red", linestyle='--')

        # fill area under plot
        head = head.append(tail.head(1))
        x1 = head.index.values
        y1 = head['volume']
        x2 = tail.index.values
        y2 = tail['volume']
        ax.fill_between(x1, y1, color="blue", alpha=0.2)
        ax.fill_between(x2, y2, color="orange", alpha=0.2)

        # create legend
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=str(items_in_head) + ': items in the head', markerfacecolor='blue', markersize=5),
                           Line2D([0], [0], marker='o', color='w', label=str(items_in_tail) + ': items in the tail', markerfacecolor='orange', markersize=5)]
        ax.legend(handles=legend_elements, loc=1)

    else:
        x1 = volume_df[item_id_column]
        y1 = volume_df['volume']
        ax.fill_between(x1, y1, color="blue", alpha=0.3)
    if not x_labels:
        plt.xticks([], [])
        ax.set(xticklabels=[])
    else:
        ax.set_xticklabels(labels=volume_df[item_id_column], rotation=45, ha="right")

    plt.savefig(save_name)

    return None


def make_cold_start_recommendation_on_new_user(model, dataset, new_user_features_df, item_ids_list, item_features=None, new_user_id=None):
    """
    A function to make cold start recommendations provided items for a new user
    :param model: The trained hybrid recommender model
    :param dataset: The dataset object, used to get the mappings between user/item ids and their internal ids.
    :param new_user_features_df: DataFrame with the user features to use.  It should be just one row with the new user features. The features must be the same as those already seen by the model
    :param item_ids_list: A list of item ids you want to consider in the predictions. If you want to run the prediction on all items, pass a list of all item ids seen by the model. You shouldn't try
                          to use this with cold start items.  Use the make_cold_start_recommendation_on_new_item function instead.
    :param item_features: The original sparse matrix with the features of the existing items in the model.
    :param new_user_id: Optional id number of the new user.  This doesn't do anything other than get added to the final output if it is given to the function.
    :return: A dataframe with the rank order scores for the new user on all required items.
    """
    # TODO: Same for cold start users and items simultaneously.  The latter is made more difficult as we'll have to deal with the internal indices pointing to specific rows of the feature matrices
    # TODO: Add more checks for inputs to this function. There are a lot of potential failure modes.

    # Convert the list of item ids to the internal indices used by the model
    internal_item_id_list = np.array([dataset.mapping()[2][id_] for id_ in item_ids_list])

    # Invert the dictionaries for reverse translation
    internal_id_to_item_dict = {val: key for key, val in dataset.mapping()[2].items()}

    item_preds_dict = {}

    # convert the input DataFrame of features for the new user into a sparse type
    new_user_features_sparse = sparse.coo_matrix(new_user_features_df.values.tolist()[0])

    # Run a prediction on all available items for the new user
    preds_array = model.predict(user_ids=[0 for id_ in internal_item_id_list], item_ids=internal_item_id_list, item_features=item_features, user_features=new_user_features_sparse)

    for idx_, item_id in enumerate(internal_item_id_list):
        item_preds_dict[item_id] = preds_array[idx_]

    # Convert to DataFrame
    item_preds_df = pd.DataFrame.from_dict(item_preds_dict, orient='index')
    item_preds_df.reset_index(inplace=True)
    item_preds_df.columns = ['item_id_internal', 'score']

    item_preds_df['item_id'] = item_preds_df['item_id_internal'].map(internal_id_to_item_dict)

    # Drop the internal ids as we don't need them anymore
    item_preds_df.drop(['item_id_internal'], axis=1, inplace=True)

    # If there is a new user id to be used here (not in the existing dataset) add it to the dataframe
    if new_user_id is not None:
        item_preds_df['user_id'] = new_user_id

    # Rank order the recommendations scores
    item_preds_df.sort_values(by='score', ascending=False, inplace=True)

    # Re-order the index after sorting
    item_preds_df.reset_index(inplace=True, drop=True)

    # Re-order the columns for returning
    if new_user_id is not None:
        item_preds_df = item_preds_df[['user_id', 'item_id', 'score']]
    else:
        item_preds_df = item_preds_df[['item_id', 'score']]

    return item_preds_df


def make_cold_start_recommendation_on_new_item(model, dataset, new_item_features_df, user_ids_list, user_features=None, new_item_id=None):
    """
    A function to make cold start recommendations on existing users for a new item
    :param model: The trained hybrid recommender model
    :param dataset: The dataset object, used to get the mappings between user/item ids and their internal ids.
    :param new_item_features_df: DataFrame with the item features to use.  It should be just one row with the new item features. The features must be the same as those already seen by the model
    :param user_ids_list: A list of user ids you want to consider in the predictions. If you want to run the prediction on all user, pass a list of all user ids seen by the model. You shouldn't try
                          to use this with cold start user.  Use the make_cold_start_recommendation_on_new_user function instead.
    :param user_features: The original sparse matrix with the features of the existing users in the model.
    :param new_item_id: Optional id number of the new item.  This doesn't do anything other than get added to the final output if it is given to the function.
    :return: A dataframe with the rank order scores for the new item on all required users.
    """
    # TODO: Same for cold start users and items simultaneously.  The latter is made more difficult as we'll have to deal with the internal indices pointing to specific rows of the feature matrices
    # TODO: Add more checks for inputs to this function. There are a lot of potential failure modes.
    # TODO: Write wrapper to get cold start to run along with hot start.

    # Convert the list of item ids to the internal indices used by the model
    internal_user_id_list = np.array([dataset.mapping()[0][id_] for id_ in user_ids_list])

    # Invert the dictionaries for reverse translation
    internal_id_to_user_dict = {val: key for key, val in dataset.mapping()[0].items()}

    user_preds_dict = {}

    # convert the input DataFrame of features for the new user into a sparse type
    new_item_features_sparse = sparse.coo_matrix(new_item_features_df.values.tolist()[0])

    # Run a prediction on all available items for the new user
    preds_array = model.predict(user_ids=internal_user_id_list, item_ids=[0 for id_ in internal_user_id_list], user_features=user_features, item_features=new_item_features_sparse)

    for idx_, user_id in enumerate(internal_user_id_list):
        user_preds_dict[user_id] = preds_array[idx_]

    # Convert to DataFrame
    user_preds_df = pd.DataFrame.from_dict(user_preds_dict, orient='index')
    user_preds_df.reset_index(inplace=True)
    user_preds_df.columns = ['user_id_internal', 'score']

    user_preds_df['user_id'] = user_preds_df['user_id_internal'].map(internal_id_to_user_dict)

    # Drop the internal ids as we don't need them anymore
    user_preds_df.drop(['user_id_internal'], axis=1, inplace=True)

    # If there is a new item id to be used here (not in the existing dataset) add it to the dataframe
    if new_item_id is not None:
        user_preds_df['item_id'] = new_item_id

    # Rank order the recommendations scores
    user_preds_df.sort_values(by='score', ascending=False, inplace=True)

    # Re-order the index after sorting
    user_preds_df.reset_index(inplace=True, drop=True)

    # Re-order the columns for returning
    if new_item_id is not None:
        user_preds_df = user_preds_df[['user_id', 'item_id', 'score']]
    else:
        user_preds_df = user_preds_df[['user_id', 'score']]

    return user_preds_df



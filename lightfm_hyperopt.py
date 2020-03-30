"""

 _     _       _     _  _________  ___     _____       _   _           _          _   _
| |   (_)     | |   | | |  ___|  \/  |    |  _  |     | | (_)         (_)        | | (_)
| |    _  __ _| |__ | |_| |_  | .  . |    | | | |_ __ | |_ _ _ __ ___  _ ______ _| |_ _  ___  _ __
| |   | |/ _` | '_ \| __|  _| | |\/| |    | | | | '_ \| __| | '_ ` _ \| |_  / _` | __| |/ _ \| '_ \
| |___| | (_| | | | | |_| |   | |  | |    \ \_/ / |_) | |_| | | | | | | |/ / (_| | |_| | (_) | | | |
\_____/_|\__, |_| |_|\__\_|   \_|  |_/     \___/| .__/ \__|_|_| |_| |_|_/___\__,_|\__|_|\___/|_| |_|
          __/ |                                 | |
         |___/                                  |_|


A custom wrapper around LightFM for optimization with random search and hyperopt.

Version 0.2.0

"""
# TODO: Test the nested cross validation function.  What's currently being done is hyperparameter estimation and then cross validation on the tune model.
# This is ok, but might lead to over-optimistic error estimates.

# Import modules

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
import dataiku

from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank

from time import gmtime, strftime

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Define the possible model weights for saving and loading
possible_model_weights = {"user_embeddings",
                          "user_biases",
                          "item_embeddings",
                          "item_biases",
                          "item_bias_momentum",
                          "item_bias_gradients",
                          "item_embedding_momentum",
                          "item_embedding_gradients",
                          "user_bias_momentum",
                          "user_bias_gradients",
                          "user_embedding_momentum",
                          "user_embedding_gradients"}


def nested_cv_dataiku(folder, interactions, hyperparams_dict, fit_params_dict, outer_valid_percentage=0.1, inner_valid_percentage=0.1, item_features=None, user_features=None, outer_cv=3, inner_cv=3, inner_seed=0, outer_seed=10, early_stop_evals=None, batch_size=None, max_evals=10, eval_metric='auc_score', k=10, random_search=False, hyper_opt_search=True, verbose=True):
    """
    Runs nested cross validation on the entire modeling process.  Recall that nested cross validation is not intended to select the optimal model/parameters, but rather provide a better estimate of
    performance due to the overall modeling process itself, of which the fitting of hyperparameters if part of. Should work in a dataiku environment
    :param folder: The dataiku folder object
    :param interactions: The full training set (sparse matrix) of user/item interactions
    :param hyperparams_dict: The dictionary of model hyperparameters.  The keys should be the hyperparameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the hyperparameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that hyperparameters
    :param fit_params_dict: The dictionary of fit parameters.  The keys should be the parameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the parameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that parameters
    :param outer_valid_percentage: The percentage of the training set you want to use for the outer loop validation
    :param inner_valid_percentage: The percentage of the outer loop training set you want to use for the inner loop validation
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :param outer_cv: The number of cross validations folds to use in the outer loop.  Should be an integer
    :param inner_cv: The number of cross validations folds to use in the inner loop.  Should be an integer
    :param random_search: True if you want to use randomized search over the parameters
    :param hyper_opt_search: True if you want to use tree parzen estimators algorithm for parameter search
    :param max_evals: The maximum number of evaluations to use for parameter search
    :param early_stop_evals: The number of evaluations that need to go by with no improvement to trigger the early stopping condition
    :param inner_seed: The random seed to use in the inner cross validation loop.  Doesn't apply to the train/test split, but should start the model training at the same place
    :param outer_seed: The random seed to use in the outer cross validation loop.  Doesn't apply to the train/test split, but should start the model training at the same place
    :param batch_size: Number of evaluations to run before dumping current best results to file
    :param eval_metric: The evaluation metric you want to use
    :param k: The k parameter for the precision at k and recall at k metrics.  Only relevant if you are using one of these metrics for valuation
    :param verbose: Controls the verbosity of the model fit.  Default is True. This means the model will print out the epoch it's on as it's training
    :return: The outer cross validation scores as a list, the list of best parameters as found by the inner cross validation loop for each outer loop iteration, a dictionary containing the results of
             each evaluation (averaged over the folds) of the inner cross validation loop indexed by the outer loop fold number
    """
    if outer_valid_percentage is None:
        raise ValueError('Please provide an outer_valid_percentage to split the input training data in the outer cross validation loop')

    if inner_valid_percentage is None:
        raise ValueError('Please provide an inner_valid_percentage to split the input training data in the inner cross validation loop')

    if inner_seed is None:
        print('The random seed is not set for the inner cross validation loop.  This will lead to potentially non-reproducible results.')
    else:
        assert type(inner_seed) == int, "inner_seed must be an integer"

    if outer_seed is not None:
        assert type(outer_seed) == int, "outer_seed must be an integer"
        outer_seed_np = np.random.RandomState(outer_seed)
    else:
        print('The random seed is not set for the outer cross validation loop.  This will lead to potentially non-reproducible results.')
        outer_seed_np = np.random.RandomState(outer_seed)

    try:
        num_threads = fit_params_dict['num_threads']
    except KeyError:
        num_threads = 1

    print('Evaluation model with {0} outer cross-validation folds and {1} inner cross-validation folds.'.format(outer_cv, inner_cv))
    print('Remember, nested cross-validation does not select the hyperparameters for the model. It estimates performance of the entire modeling processes.')

    if early_stop_evals is not None:
        if batch_size is not None:
            raise AttributeError('Cannot use both early stopping and batch fitting simultaneously. Please use one or the other.')

    if hyper_opt_search:
        if random_search:
            raise AttributeError('Cannot run random search and hyperopt search simultaneously. Please select one of the other.')

    outer_score_list = []
    param_list = []
    inner_fold_results = {}
    # outer cv loop
    for outer_fold in range(outer_cv):
        train_, outer_valid_ = random_train_test_split(interactions=interactions, test_percentage=outer_valid_percentage)

        # Now run the hyperparameter search here.  This should return the best found hyperparameters based on inner cv score.
        if early_stop_evals is not None:
            best, trials = fit_model_early_stopping_dataiku(folder=folder,
                                                            interactions=train_,
                                                            hyperparams_dict=hyperparams_dict,
                                                            fit_params_dict=fit_params_dict,
                                                            test_percentage=inner_valid_percentage,
                                                            item_features=item_features,
                                                            user_features=user_features,
                                                            cv=inner_cv,
                                                            max_evals=max_evals,
                                                            eval_metric=eval_metric,
                                                            verbose=verbose,
                                                            seed=inner_seed,
                                                            k=k,
                                                            random_search=random_search,
                                                            hyper_opt_search=hyper_opt_search,
                                                            early_stop_evals=early_stop_evals)
        elif batch_size is not None:
            best, trials = fit_model_batch_dataiku(folder=folder,
                                                   interactions=train_,
                                                   hyperparams_dict=hyperparams_dict,
                                                   fit_params_dict=fit_params_dict,
                                                   test_percentage=inner_valid_percentage,
                                                   item_features=item_features,
                                                   user_features=user_features,
                                                   cv=inner_cv,
                                                   max_evals=max_evals,
                                                   eval_metric=eval_metric,
                                                   verbose=verbose,
                                                   seed=inner_seed,
                                                   k=k,
                                                   random_search=random_search,
                                                   hyper_opt_search=hyper_opt_search,
                                                   batch_size=batch_size)
        else:
            best, trials = fit_model(interactions=train_,
                                     hyperparams_dict=hyperparams_dict,
                                     fit_params_dict=fit_params_dict,
                                     test_percentage=inner_valid_percentage,
                                     item_features=item_features,
                                     user_features=user_features,
                                     cv=inner_cv,
                                     max_evals=max_evals,
                                     eval_metric=eval_metric,
                                     verbose=verbose,
                                     seed=inner_seed,
                                     k=k,
                                     random_search=random_search,
                                     hyper_opt_search=hyper_opt_search)

        # Store the results of the trials object for this inner cv loop.
        # This will store the results for each fold in the most recent inner fold to a dictionary indexed by the current outer fold number
        inner_fold_results[outer_fold] = trials.trials

        # Get out the best hyperparameters. Should I support saving the parameters here? Doesn't seem like much point.
        inner_params = get_best_hyperparams(hyperparams_dict=hyperparams_dict,
                                            fit_params_dict=fit_params_dict,
                                            best=best)

        # Add the random seed to the inner_params dictionary. We need this if the user wants to specify the seed in the outer cross validation fit as well.
        inner_params['random_state'] = outer_seed_np

        # param_list.append(inner_params)
        num_epochs = inner_params.pop('num_epochs')

        # Train model on train_ with the returned hyperparameters and test on outer_valid_
        score = fit_eval(params=inner_params,
                         eval_metric=eval_metric,
                         train_interactions=train_,
                         valid_interactions=outer_valid_,
                         num_epochs=num_epochs,
                         num_threads=num_threads,
                         item_features=item_features,
                         user_features=user_features,
                         k=k,
                         verbose=verbose)

        # Add the number of epochs back to the parameter dictionary before storing it. We might want to look at this later
        inner_params['num_epochs'] = num_epochs

        # Add the seed used in the inner loop in case someone wants to set separate inner and out seeds just to see what happens
        inner_params['inner_cv_seed'] = inner_seed

        # Rename in the outer cv seed to something more interpretable in the output. Remove the numpy random state and just output the initially input integer
        inner_params.pop('random_state')
        inner_params['outer_cv_seed'] = outer_seed

        param_list.append(inner_params)

        outer_score_list.append(score)

        print('Completed outer fold: {0}'.format(outer_fold + 1))
        print('Outer fold: {0}  loss: {1}'.format(outer_fold + 1, score))

    print('Nested cross-validation complete')

    return outer_score_list, param_list, inner_fold_results


def nested_cv(interactions, hyperparams_dict, fit_params_dict, outer_valid_percentage=0.1, inner_valid_percentage=0.1, item_features=None, user_features=None, outer_cv=3, inner_cv=3, inner_seed=0, outer_seed=10, early_stop_evals=None, batch_size=None, max_evals=10, eval_metric='auc_score', k=10, random_search=False, hyper_opt_search=True, verbose=True):
    """
    Runs nested cross validation on the entire modeling process.  Recall that nested cross validation is not intended to select the optimal model/parameters, but rather provide a better estimate of
    performance due to the overall modeling process itself, of which the fitting of hyperparameters if part of.
    :param interactions: The full training set (sparse matrix) of user/item interactions
    :param hyperparams_dict: The dictionary of model hyperparameters.  The keys should be the hyperparameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the hyperparameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that hyperparameters
    :param fit_params_dict: The dictionary of fit parameters.  The keys should be the parameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the parameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that parameters
    :param outer_valid_percentage: The percentage of the training set you want to use for the outer loop validation
    :param inner_valid_percentage: The percentage of the outer loop training set you want to use for the inner loop validation
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :param outer_cv: The number of cross validations folds to use in the outer loop.  Should be an integer
    :param inner_cv: The number of cross validations folds to use in the inner loop.  Should be an integer
    :param random_search: True if you want to use randomized search over the parameters
    :param hyper_opt_search: True if you want to use tree parzen estimators algorithm for parameter search
    :param max_evals: The maximum number of evaluations to use for parameter search
    :param early_stop_evals: The number of evaluations that need to go by with no improvement to trigger the early stopping condition
    :param inner_seed: The random seed to use in the inner cross validation loop.  Doesn't apply to the train/test split, but should start the model training at the same place
    :param outer_seed: The random seed to use in the outer cross validation loop.  Doesn't apply to the train/test split, but should start the model training at the same place
    :param batch_size: Number of evaluations to run before dumping current best results to file
    :param eval_metric: The evaluation metric you want to use
    :param k: The k parameter for the precision at k and recall at k metrics.  Only relevant if you are using one of these metrics for valuation
    :param verbose: Controls the verbosity of the model fit.  Default is True. This means the model will print out the epoch it's on as it's training
    :return: The outer cross validation scores as a list, the list of best parameters as found by the inner cross validation loop for each outer loop iteration, a dictionary containing the results of
             each evaluation (averaged over the folds) of the inner cross validation loop indexed by the outer loop fold number
    """
    if outer_valid_percentage is None:
        raise ValueError('Please provide an outer_valid_percentage to split the input training data in the outer cross validation loop')

    if inner_valid_percentage is None:
        raise ValueError('Please provide an inner_valid_percentage to split the input training data in the inner cross validation loop')

    if inner_seed is None:
        print('The random seed is not set for the inner cross validation loop.  This will lead to potentially non-reproducible results.')
    else:
        assert type(inner_seed) == int, "inner_seed must be an integer"

    if outer_seed is not None:
        assert type(outer_seed) == int, "outer_seed must be an integer"
        outer_seed_np = np.random.RandomState(outer_seed)
    else:
        print('The random seed is not set for the outer cross validation loop.  This will lead to potentially non-reproducible results.')
        outer_seed_np = np.random.RandomState(outer_seed)

    try:
        num_threads = fit_params_dict['num_threads']
    except KeyError:
        num_threads = 1

    print('Evaluation model with {0} outer cross-validation folds and {1} inner cross-validation folds.'.format(outer_cv, inner_cv))
    print('Remember, nested cross-validation does not select the hyperparameters for the model. It estimates performance of the entire modeling processes.')

    if early_stop_evals is not None:
        if batch_size is not None:
            raise AttributeError('Cannot use both early stopping and batch fitting simultaneously. Please use one or the other.')

    if hyper_opt_search:
        if random_search:
            raise AttributeError('Cannot run random search and hyperopt search simultaneously. Please select one of the other.')

    outer_score_list = []
    param_list = []
    inner_fold_results = {}
    # outer cv loop
    for outer_fold in range(outer_cv):
        train_, outer_valid_ = random_train_test_split(interactions=interactions, test_percentage=outer_valid_percentage)

        # Now run the hyperparameter search here.  This should return the best found hyperparameters based on inner cv score.
        if early_stop_evals is not None:
            best, trials = fit_model_early_stopping(interactions=train_,
                                                    hyperparams_dict=hyperparams_dict,
                                                    fit_params_dict=fit_params_dict,
                                                    test_percentage=inner_valid_percentage,
                                                    item_features=item_features,
                                                    user_features=user_features,
                                                    cv=inner_cv,
                                                    max_evals=max_evals,
                                                    eval_metric=eval_metric,
                                                    verbose=verbose,
                                                    seed=inner_seed,
                                                    k=k,
                                                    random_search=random_search,
                                                    hyper_opt_search=hyper_opt_search,
                                                    early_stop_evals=early_stop_evals)
        elif batch_size is not None:
            best, trials = fit_model_batch(interactions=train_,
                                           hyperparams_dict=hyperparams_dict,
                                           fit_params_dict=fit_params_dict,
                                           test_percentage=inner_valid_percentage,
                                           item_features=item_features,
                                           user_features=user_features,
                                           cv=inner_cv,
                                           max_evals=max_evals,
                                           eval_metric=eval_metric,
                                           verbose=verbose,
                                           seed=inner_seed,
                                           k=k,
                                           random_search=random_search,
                                           hyper_opt_search=hyper_opt_search,
                                           batch_size=batch_size)
        else:
            best, trials = fit_model(interactions=train_,
                                     hyperparams_dict=hyperparams_dict,
                                     fit_params_dict=fit_params_dict,
                                     test_percentage=inner_valid_percentage,
                                     item_features=item_features,
                                     user_features=user_features,
                                     cv=inner_cv,
                                     max_evals=max_evals,
                                     eval_metric=eval_metric,
                                     verbose=verbose,
                                     seed=inner_seed,
                                     k=k,
                                     random_search=random_search,
                                     hyper_opt_search=hyper_opt_search)

        # Store the results of the trials object for this inner cv loop.
        # This will store the results for each fold in the most recent inner fold to a dictionary indexed by the current outer fold number
        inner_fold_results[outer_fold] = trials.trials

        # Get out the best hyperparameters. Should I support saving the parameters here? Doesn't seem like much point.
        inner_params = get_best_hyperparams(hyperparams_dict=hyperparams_dict,
                                            fit_params_dict=fit_params_dict,
                                            best=best)

        # Add the random seed to the inner_params dictionary. We need this if the user wants to specify the seed in the outer cross validation fit as well.
        inner_params['random_state'] = outer_seed_np

        # param_list.append(inner_params)
        num_epochs = inner_params.pop('num_epochs')

        # Train model on train_ with the returned hyperparameters and test on outer_valid_
        score = fit_eval(params=inner_params,
                         eval_metric=eval_metric,
                         train_interactions=train_,
                         valid_interactions=outer_valid_,
                         num_epochs=num_epochs,
                         num_threads=num_threads,
                         item_features=item_features,
                         user_features=user_features,
                         k=k,
                         verbose=verbose)

        # Add the number of epochs back to the parameter dictionary before storing it. We might want to look at this later
        inner_params['num_epochs'] = num_epochs

        # Add the seed used in the inner loop in case someone wants to set separate inner and out seeds just to see what happens
        inner_params['inner_cv_seed'] = inner_seed

        # Rename in the outer cv seed to something more interpretable in the output. Remove the numpy random state and just output the initially input integer
        inner_params.pop('random_state')
        inner_params['outer_cv_seed'] = outer_seed

        param_list.append(inner_params)

        outer_score_list.append(score)

        print('Completed outer fold: {0}'.format(outer_fold + 1))
        print('Outer fold: {0}  loss: {1}'.format(outer_fold + 1, score))

    print('Nested cross-validation complete')

    return outer_score_list, param_list, inner_fold_results


def fit_model_early_stopping(interactions, hyperparams_dict, fit_params_dict, test_percentage=0.1, item_features=None, user_features=None, cv=None, random_search=False, hyper_opt_search=True, max_evals=10, early_stop_evals=10, seed=0, eval_metric='auc_score', k=10, verbose=True):
    """
    Higher level function to actually run all the aspects of the hyperparameter search. This one works with early stopping; it will end the parameter search when no performance improvement has been
    detected after early_stop_evals threshold number of evaluations.
    :param interactions: The full training set (sparse matrix) of user/item interactions
    :param hyperparams_dict: The dictionary of model hyperparameters.  The keys should be the hyperparameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the hyperparameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that hyperparameters
    :param fit_params_dict: The dictionary of fit parameters.  The keys should be the parameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the parameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that parameters
    :param test_percentage: The percentage of the training set you want to use for validation
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :param cv: The number of cross validation folds to use.  Should be an interger number of folds or None if you don't want to run with cross validation
    :param random_search: True if you want to use randomized search over the parameters
    :param hyper_opt_search: True if you want to use tree parzen estimators algorithm for parameter search
    :param max_evals: The maximum number of evaluations to use for parameter search
    :param early_stop_evals: The number of evaluations that need to go by with no improvement to trigger the early stopping condition
    :param seed: The random seed to use in model building.  Doesn't apply to the train/test split, but should start the model training at the same place
    :param eval_metric: The evaluation metric to use
    :param k: The k parameter for the precision at k and recall at k metrics.  Only relevant if you are using one of these metrics for valuation
    :param verbose: Controls the verbosity of the model fit.  Default is True. This means the model will print out the epoch it's on as it's training
    :return: The best found parameters and the history trials object
    """
    # TODO: I don't think it makes a lot of sense to have yet another function to do this.  Merge this with the batch fitting function.
    if random_search:
        if not hyper_opt_search:
            print('Running randomized hyperparameter search')
        else:
            print('Please select either random search or hyperopt search')
            return None

    if hyper_opt_search:
        if not random_search:
            print('Running hyperopt hyperparameter search')
        else:
            print('Please select either random search or hyperopt search')
            return None

    params = prep_params_for_hyperopt(hyperparams_dict=hyperparams_dict,
                                      fit_params_dict=fit_params_dict,
                                      interactions=interactions,
                                      test_percentage=test_percentage,
                                      item_features=item_features,
                                      user_features=user_features,
                                      cv=cv,
                                      eval_metric=eval_metric,
                                      k=k,
                                      verbose=verbose)
    trials = Trials()

    if cv is not None:
        print('Running in cross validation mode for {0} folds'.format(cv))
    else:
        print('Not running in cross validation mode. Will default to single train test split')

    if random_search:
        # Run for early stopping
        print('Running for {0} evaluations with early stopping checks every {1} evaluations'.format(max_evals, early_stop_evals))
        best_loss_so_far = 0
        for i in range(early_stop_evals, max_evals + 1, early_stop_evals):

            best = fmin(f_objective, params, algo=tpe.rand.suggest, max_evals=i, trials=trials, rstate=np.random.RandomState(seed))

            # Output the batch findings so far
            file_name_trials = output_checkpoint_files(hyperparams_dict, fit_params_dict, i, trials, best)
            # Reload the trials object to restart the search where we left off
            # TODO: Do I need to do this here, or can I just re-assign the variable?
            trials = pickle.load(open(file_name_trials, "rb"))

            current_loss = trials.best_trial['result']['loss']

            if current_loss < best_loss_so_far:
                best_loss_so_far = current_loss
                print('Loss has improved. Continuing with search')
            else:
                print('No loss improvement after {0} evaluations. This has triggered the early stopping criterion.'.format(early_stop_evals))
                return best, trials

        return best, trials

    if hyper_opt_search:
        # Run for early stopping
        print('Running for {0} evaluations with early stopping checks every {1} evaluations'.format(max_evals, early_stop_evals))
        best_loss_so_far = 0
        for i in range(early_stop_evals, max_evals + 1, early_stop_evals):

            best = fmin(f_objective, params, algo=tpe.suggest, max_evals=i, trials=trials, rstate=np.random.RandomState(seed))

            # Output the batch findings so far
            file_name_trials = output_checkpoint_files(hyperparams_dict, fit_params_dict, i, trials, best)
            # Reload the trials object to restart the search where we left off
            # TODO: Do I need to do this here, or can I just re-assign the variable?
            trials = pickle.load(open(file_name_trials, "rb"))

            current_loss = trials.best_trial['result']['loss']

            if current_loss < best_loss_so_far:
                best_loss_so_far = current_loss
                print('Loss has improved. Continuing with search')
            else:
                print('No loss improvement after {0} evaluations. This has triggered the early stopping criterion.'.format(early_stop_evals))
                return best, trials

        return best, trials


def fit_model_early_stopping_dataiku(folder, interactions, hyperparams_dict, fit_params_dict, test_percentage=0.1, item_features=None, user_features=None, cv=None, random_search=False, hyper_opt_search=True, max_evals=10, early_stop_evals=10, seed=0, eval_metric='auc_score', k=10, verbose=True):
    """
    Higher level function to actually run all the aspects of the hyperparameter search. This one works with early stopping; it will end the parameter search when no performance improvement has been
    detected after early_stop_evals threshold number of evaluations. Should work in a dataiku environment
    :param folder: The dataiku folder object
    :param interactions: The full training set (sparse matrix) of user/item interactions
    :param hyperparams_dict: The dictionary of model hyperparameters.  The keys should be the hyperparameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the hyperparameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that hyperparameters
    :param fit_params_dict: The dictionary of fit parameters.  The keys should be the parameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the parameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that parameters
    :param test_percentage: The percentage of the training set you want to use for validation
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :param cv: The number of cross validation folds to use.  Should be an interger number of folds or None if you don't want to run with cross validation
    :param random_search: True if you want to use randomized search over the parameters
    :param hyper_opt_search: True if you want to use tree parzen estimators algorithm for parameter search
    :param max_evals: The maximum number of evaluations to use for parameter search
    :param early_stop_evals: The number of evaluations that need to go by with no improvement to trigger the early stopping condition
    :param seed: The random seed to use in model building.  Doesn't apply to the train/test split, but should start the model training at the same place
    :param eval_metric: The evaluation metric to use
    :param k: The k parameter for the precision at k and recall at k metrics.  Only relevant if you are using one of these metrics for valuation
    :param verbose: Controls the verbosity of the model fit.  Default is True. This means the model will print out the epoch it's on as it's training
    :return: The best found parameters and the history trials object
    """
    # TODO: I don't think it makes a lot of sense to have yet another function to do this.  Merge this with the batch fitting function.
    if random_search:
        if not hyper_opt_search:
            print('Running randomized hyperparameter search')
        else:
            print('Please select either random search or hyperopt search')
            return None

    if hyper_opt_search:
        if not random_search:
            print('Running hyperopt hyperparameter search')
        else:
            print('Please select either random search or hyperopt search')
            return None

    params = prep_params_for_hyperopt(hyperparams_dict=hyperparams_dict,
                                      fit_params_dict=fit_params_dict,
                                      interactions=interactions,
                                      test_percentage=test_percentage,
                                      item_features=item_features,
                                      user_features=user_features,
                                      cv=cv,
                                      eval_metric=eval_metric,
                                      k=k,
                                      verbose=verbose)
    trials = Trials()

    if cv is not None:
        print('Running in cross validation mode for {0} folds'.format(cv))
    else:
        print('Not running in cross validation mode. Will default to single train test split')

    if random_search:
        # Run for early stopping
        print('Running for {0} evaluations with early stopping checks every {1} evaluations'.format(max_evals, early_stop_evals))
        best_loss_so_far = 0
        for i in range(early_stop_evals, max_evals + 1, early_stop_evals):

            best = fmin(f_objective, params, algo=tpe.rand.suggest, max_evals=i, trials=trials, rstate=np.random.RandomState(seed))

            # Output the batch findings so far
            file_name_trials = output_checkpoint_files_dataiku(folder, hyperparams_dict, fit_params_dict, i, trials, best)

            # Setup the dataiku handle object
            handle = dataiku.Folder(folder)
            path = handle.get_path()

            # Reload the trials object to restart the search where we left off
            # TODO: Do I need to to do here, or can I just re-assign the variable?
            trials = pickle.load(open(path + "/" + file_name_trials, "rb", -1))

            current_loss = trials.best_trial['result']['loss']

            if current_loss < best_loss_so_far:
                best_loss_so_far = current_loss
                print('Loss has improved. Continuing with search')
            else:
                print('No loss improvement after {0} evaluations. This has triggered the early stopping criterion.'.format(early_stop_evals))
                return best, trials

        return best, trials

    if hyper_opt_search:
        # Run for early stopping
        print('Running for {0} evaluations with early stopping checks every {1} evaluations'.format(max_evals, early_stop_evals))
        best_loss_so_far = 0
        for i in range(early_stop_evals, max_evals + 1, early_stop_evals):

            best = fmin(f_objective, params, algo=tpe.suggest, max_evals=i, trials=trials, rstate=np.random.RandomState(seed))

            # Output the batch findings so far
            file_name_trials = output_checkpoint_files_dataiku(folder, hyperparams_dict, fit_params_dict, i, trials, best)

            # Setup the dataiku handle object
            handle = dataiku.Folder(folder)
            path = handle.get_path()

            # Reload the trials object to restart the search where we left off
            # TODO: Do I need to to do here, or can I just re-assign the variable?
            trials = pickle.load(open(path + "/" + file_name_trials, "rb", -1))

            current_loss = trials.best_trial['result']['loss']

            if current_loss < best_loss_so_far:
                best_loss_so_far = current_loss
                print('Loss has improved. Continuing with search')
            else:
                print('No loss improvement after {0} evaluations. This has triggered the early stopping criterion.'.format(early_stop_evals))
                return best, trials

        return best, trials


def fit_model_batch(interactions, hyperparams_dict, fit_params_dict, batch_size, test_percentage=0.1, item_features=None, user_features=None, cv=None, random_search=False, hyper_opt_search=True, max_evals=10, seed=0, eval_metric='auc_score', k=10, verbose=True):
    """
    Higher level function to actually run all the aspects of the hyperparameter search. This version works with batches. The hyperparameter search will occur as per usual, but the results will be
    logged every batch_size number of evaluations and restarted from there.
    :param interactions: The full training set (sparse matrix) of user/item interactions
    :param hyperparams_dict: The dictionary of model hyperparameters.  The keys should be the hyperparameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the hyperparameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that hyperparameters
    :param fit_params_dict: The dictionary of fit parameters.  The keys should be the parameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the parameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that parameters
    :param batch_size: Number of evaluations to run before dumping current best results to file
    :param test_percentage: The percentage of the training set you want to use for validation
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :param cv: The number of cross validation folds to use.  Should be an integer number of folds or None if you don't want to run with cross validation
    :param random_search: True if you want to use randomized search over the parameters
    :param hyper_opt_search: True if you want to use tree parzen estimators algorithm for parameter search
    :param max_evals: The maximum number of evaluations to use for parameter search
    :param seed: The random seed to use in model building.  Doesn't apply to the train/test split, but should start the model training at the same place
    :param eval_metric: The evaluation metric to use
    :param k: The k parameter for the precision at k and recall at k metrics.  Only relevant if you are using one of these metrics for valuation
    :param verbose: Controls the verbosity of the model fit.  Default is True. This means the model will print out the epoch it's on as it's training
    :return: The best found parameters and the history trials object
    """
    # TODO: Throw a warning that the number of batches should divide the total number of evals, or else all evals will not get completed. e.g. batch_size=10 and max_evals=11 will only run 10 evals
    if random_search:
        if not hyper_opt_search:
            print('Running randomized hyperparameter search')
        else:
            print('Please select either random search or hyperopt search')
            return None

    if hyper_opt_search:
        if not random_search:
            print('Running hyperopt hyperparameter search')
        else:
            print('Please select either random search or hyperopt search')
            return None

    params = prep_params_for_hyperopt(hyperparams_dict=hyperparams_dict,
                                      fit_params_dict=fit_params_dict,
                                      interactions=interactions,
                                      test_percentage=test_percentage,
                                      item_features=item_features,
                                      user_features=user_features,
                                      cv=cv,
                                      eval_metric=eval_metric,
                                      k=k,
                                      verbose=verbose)
    trials = Trials()

    if cv is not None:
        print('Running in cross validation mode for {0} folds'.format(cv))
    else:
        print('Not running in cross validation mode. Will default to single train test split')

    if random_search:
        # Iterate over the maximum evaluations by batch size
        # Careful here, this will cutoff the evaluations if max_evals / batch_size is not an integer...so max_evals = 11 will only run one batch. It still does 11 evals.
        print('Running for {0} evaluations in batches of {1} for a total of {2} batches'.format(max_evals, batch_size, max_evals // batch_size))
        for i in range(batch_size, max_evals + 1, batch_size):

            best = fmin(f_objective, params, algo=tpe.rand.suggest, max_evals=i, trials=trials, rstate=np.random.RandomState(seed))

            # Output the batch findings so far
            file_name_trials = output_checkpoint_files(hyperparams_dict, fit_params_dict, i, trials, best)

            # Reload the trials object to restart the search where we left off
            # TODO: Do I need to to do here, or can I just re-assign the variable?
            trials = pickle.load(open(file_name_trials, "rb"))

            batch_number = int(max_evals * (i / max_evals) / batch_size)
            print('completed batch: {0}'.format(batch_number))
            print('completed evaluations: {0} to {1}'.format(i - batch_size, i))

        return best, trials

    if hyper_opt_search:
        # Iterate over the maximum evaluations by batch size
        # Careful here, this will cutoff the evaluations if max_evals / batch_size is not an integer...so max_evals = 11 will only run one batch. It still does 11 evals.
        print('Running for {0} evaluations in batches of {1} for a total of {2} batches'.format(max_evals, batch_size, max_evals // batch_size))
        for i in range(batch_size, max_evals + 1, batch_size):

            best = fmin(f_objective, params, algo=tpe.suggest, max_evals=i, trials=trials, rstate=np.random.RandomState(seed))

            # Output the batch findings so far
            file_name_trials = output_checkpoint_files(hyperparams_dict, fit_params_dict, i, trials, best)
            # Reload the trials object to restart the search where we left off
            # TODO: Do I need to to do here, or can I just re-assign the variable?
            trials = pickle.load(open(file_name_trials, "rb"))

            batch_number = int(max_evals * (i / max_evals) / batch_size)
            print('completed batch: {0}'.format(batch_number))
            print('completed evaluations: {0} to {1}'.format(i - batch_size, i))

        return best, trials


def fit_model_batch_dataiku(folder, interactions, hyperparams_dict, fit_params_dict, batch_size, test_percentage=0.1, item_features=None, user_features=None, cv=None, random_search=False, hyper_opt_search=True, max_evals=10, seed=0, eval_metric='auc_score', k=10, verbose=True):
    """
    Higher level function to actually run all the aspects of the hyperparameter search. This version works with batches. The hyperparameter search will occur as per usual, but the results will be
    logged every batch_size number of evaluations and restarted from there. Should work in a dataiku environment.
    :param folder: The dataiku folder object
    :param interactions: The full training set (sparse matrix) of user/item interactions
    :param hyperparams_dict: The dictionary of model hyperparameters.  The keys should be the hyperparameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the hyperparameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that hyperparameters
    :param fit_params_dict: The dictionary of fit parameters.  The keys should be the parameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the parameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that parameters
    :param batch_size: Number of evaluations to run before dumping current best results to file
    :param test_percentage: The percentage of the training set you want to use for validation
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :param cv: The number of cross validation folds to use.  Should be an integer number of folds or None if you don't want to run with cross validation
    :param random_search: True if you want to use randomized search over the parameters
    :param hyper_opt_search: True if you want to use tree parzen estimators algorithm for parameter search
    :param max_evals: The maximum number of evaluations to use for parameter search
    :param seed: The random seed to use in model building.  Doesn't apply to the train/test split, but should start the model training at the same place
    :param eval_metric: The evaluation metric to use
    :param k: The k parameter for the precision at k and recall at k metrics.  Only relevant if you are using one of these metrics for valuation
    :param verbose: Controls the verbosity of the model fit.  Default is True. This means the model will print out the epoch it's on as it's training
    :return: The best found parameters and the history trials object
    """
    # TODO: Throw a warning that the number of batches should divide the total number of evals, or else all evals will not get completed. e.g. batch_size=10 and max_evals=11 will only run 10 evals
    if random_search:
        if not hyper_opt_search:
            print('Running randomized hyperparameter search')
        else:
            print('Please select either random search or hyperopt search')
            return None

    if hyper_opt_search:
        if not random_search:
            print('Running hyperopt hyperparameter search')
        else:
            print('Please select either random search or hyperopt search')
            return None

    params = prep_params_for_hyperopt(hyperparams_dict=hyperparams_dict,
                                      fit_params_dict=fit_params_dict,
                                      interactions=interactions,
                                      test_percentage=test_percentage,
                                      item_features=item_features,
                                      user_features=user_features,
                                      cv=cv,
                                      eval_metric=eval_metric,
                                      k=k,
                                      verbose=verbose)
    trials = Trials()

    if cv is not None:
        print('Running in cross validation mode for {0} folds'.format(cv))
    else:
        print('Not running in cross validation mode. Will default to single train test split')

    if random_search:
        # Iterate over the maximum evaluations by batch size
        # Careful here, this will cutoff the evaluations if max_evals / batch_size is not an integer...so max_evals = 11 will only run one batch. It still does 11 evals.
        print('Running for {0} evaluations in batches of {1} for a total of {2} batches'.format(max_evals, batch_size, max_evals // batch_size))
        for i in range(batch_size, max_evals + 1, batch_size):

            best = fmin(f_objective, params, algo=tpe.rand.suggest, max_evals=i, trials=trials, rstate=np.random.RandomState(seed))

            # Output the batch findings so far
            file_name_trials = output_checkpoint_files_dataiku(folder, hyperparams_dict, fit_params_dict, i, trials, best)

            # Setup the dataiku handle object
            handle = dataiku.Folder(folder)
            path = handle.get_path()

            # Reload the trials object to restart the search where we left off
            # TODO: Do I need to to do here, or can I just re-assign the variable?
            trials = pickle.load(open(path + "/" + file_name_trials, "rb", -1))

            batch_number = int(max_evals * (i / max_evals) / batch_size)
            print('completed batch: {0}'.format(batch_number))
            print('completed evaluations: {0} to {1}'.format(i - batch_size, i))

        return best, trials

    if hyper_opt_search:
        # Iterate over the maximum evaluations by batch size
        # Careful here, this will cutoff the evaluations if max_evals / batch_size is not an integer...so max_evals = 11 will only run one batch. It still does 11 evals.
        print('Running for {0} evaluations in batches of {1} for a total of {2} batches'.format(max_evals, batch_size, max_evals // batch_size))
        for i in range(batch_size, max_evals + 1, batch_size):

            best = fmin(f_objective, params, algo=tpe.suggest, max_evals=i, trials=trials, rstate=np.random.RandomState(seed))

            # Output the batch findings so far
            file_name_trials = output_checkpoint_files_dataiku(folder, hyperparams_dict, fit_params_dict, i, trials, best)

            # Setup the dataiku handle object
            handle = dataiku.Folder(folder)
            path = handle.get_path()

            # Reload the trials object to restart the search where we left off
            # TODO: Do I need to to do here, or can I just re-assign the variable?
            trials = pickle.load(open(path + "/" + file_name_trials, "rb", -1))

            batch_number = int(max_evals * (i / max_evals) / batch_size)
            print('completed batch: {0}'.format(batch_number))
            print('completed evaluations: {0} to {1}'.format(i - batch_size, i))

        return best, trials


def fit_model(interactions, hyperparams_dict, fit_params_dict, test_percentage=0.1, item_features=None, user_features=None, cv=None, random_search=False, hyper_opt_search=True, max_evals=10, seed=0, eval_metric='auc_score', k=10, verbose=True):
    """
    Higher level function to actually run all the aspects of the hyperparameter search.
    :param interactions: The full training set (sparse matrix) of user/item interactions
    :param hyperparams_dict: The dictionary of model hyperparameters.  The keys should be the hyperparameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the hyperparameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that hyperparameters
    :param fit_params_dict: The dictionary of fit parameters.  The keys should be the parameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the parameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that parameters
    :param test_percentage: The percentage of the training set you want to use for validation
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :param cv: The number of cross validation folds to use.  Should be an integer number of folds or None if you don't want to run with cross validation
    :param random_search: True if you want to use randomized search over the parameters
    :param hyper_opt_search: True if you want to use tree parzen estimators algorithm for parameter search
    :param max_evals: The maximum number of evaluations to use for parameter search
    :param seed: The random seed to use in model building.  Doesn't apply to the train/test split, but should start the model training at the same place
    :param eval_metric: The evaluation metric to use
    :param k: The k parameter for the precision at k and recall at k metrics.  Only relevant if you are using one of these metrics for valuation
    :param verbose: Controls the verbosity of the model fit.  Default is True. This means the model will print out the epoch it's on as it's training
    :return: The best found parameters and the history trials object
    """
    if random_search:
        if not hyper_opt_search:
            print('Running randomized hyperparameter search')
        else:
            print('Please select either random search or hyperopt search')
            return None

    if hyper_opt_search:
        if not random_search:
            print('Running hyperopt hyperparameter search')
        else:
            print('Please select either random search or hyperopt search')
            return None

    params = prep_params_for_hyperopt(hyperparams_dict=hyperparams_dict,
                                      fit_params_dict=fit_params_dict,
                                      interactions=interactions,
                                      test_percentage=test_percentage,
                                      item_features=item_features,
                                      user_features=user_features,
                                      cv=cv,
                                      eval_metric=eval_metric,
                                      k=k,
                                      verbose=verbose)
    trials = Trials()

    if cv is not None:
        print('Running in cross validation mode for {0} folds'.format(cv))
    else:
        print('Not running in cross validation mode. Will default to single train test split')

    if random_search:
        best = fmin(f_objective, params, algo=tpe.rand.suggest, max_evals=max_evals, trials=trials, rstate=np.random.RandomState(seed))

        return best, trials

    if hyper_opt_search:
        best = fmin(f_objective, params, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.RandomState(seed))

        return best, trials


def prep_params_for_hyperopt(hyperparams_dict, fit_params_dict, interactions, test_percentage, item_features, user_features, cv, eval_metric, k, verbose):
    """
    Formats the input range of hyperparameters and fit parameters to search over for model optimization for use in HyperOpt
    :param hyperparams_dict: The dictionary of model hyperparameters.  The keys should be the hyperparameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the hyperparameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that hyperparameters
    :param fit_params_dict: The dictionary of fit parameters.  The keys should be the parameter name and the values a list with the first element the HyperOpt variable type and the second
                             element should be a list of possible values to consider for the parameter.  If only a key and number value are provided, it is assumed that the variable is a choice
                             type and if the only number you want to consider in your model optimization for that parameters
    :param interactions: The full user/item interactions training set (sparse matrix) to build the model on
    :param test_percentage: The percentage of the training set you want to use for validation
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :param cv: The number of cross validation folds to use. Should be an integer number of folds, or None if you don't want to run with cross validation
    :param eval_metric: The evaluation metric to use
    :param k: The k parameter for the precision at k and recall at k metrics.  Only relevant if you are using one of these metrics for valuation
    :param verbose: Controls the verbosity of the model fit.  Default is True. This means the model will print out the epoch it's on as it's training
    :return: The parameter grid formated to work with HyperOpt
    """
    params = {}

    # Parse the hyperparameters dictionary
    for key, val in hyperparams_dict.items():
        if isinstance(val, list):

            # First assign the proper format for argument inputs
            if val[0] == 'choice':
                arg_ = [key] + val[1::]
            else:
                arg_ = [key] + val[1::][0]

            if val[0] == 'choice':
                params[key] = hp.choice(*arg_)
            elif val[0] == 'uniform':
                params[key] = hp.uniform(*arg_)
            elif val[0] == 'quniform':
                params[key] = hp.quniform(*arg_)
            elif val[0] == 'loguniform':
                params[key] = hp.loguniform(*arg_)
            elif val[0] == 'qloguniform':
                params[key] = hp.qloguniform(*arg_)
            elif val[0] == 'normal':
                params[key] = hp.normal(*arg_)
            elif val[0] == 'qnormal':
                params[key] = hp.qnormal(*arg_)
            elif val[0] == 'lognormal':
                params[key] = hp.lognormal(*arg_)
            elif val[0] == 'qlognormal':
                params[key] = hp.qlognormal(*arg_)
            elif val[0] == 'randint':
                params[key] = hp.randint(*arg_)
            else:
                raise ValueError('Unsupported hyperparameter distribution {0} provided'.format(val[0]))
        else:
            # if not list is provided, assume a choice parameter
            arg_ = [key] + [[val]]
            params[key] = hp.randint(*arg_)

    # Parse the fit parameters dictionary
    for key, val in fit_params_dict.items():
        # These should only be discrete choice distributions
        if isinstance(val, list):
            arg_ = [key] + val[1::]
        else:
            arg_ = [key] + [[val]]
        params[key] = hp.choice(*arg_)

    # Parse the interactions data
    params['data'] = hp.choice('data', [interactions])

    # Parse the test_percentage
    params['test_percentage'] = hp.choice('test_percentage', [test_percentage])

    # Parse the item_features
    params['item_features'] = hp.choice('item_features', [item_features])

    # Parse user_features
    params['user_features'] = hp.choice('user_features', [user_features])

    # Parse cv
    params['cv'] = hp.choice('cv', [cv])

    # Parse evaluation metric
    params['eval_metric'] = hp.choice('eval_metric', [eval_metric])

    # Parse k
    params['k'] = hp.choice('k', [k])

    # Parse verbosity flag
    params['verbose'] = hp.choice('verbose', [verbose])

    return params


def get_best_hyperparams(hyperparams_dict, fit_params_dict, best, file_name=None):
    """
    Helper function to extract the numerical values of best hyperparameters from hyperopt into a more easily usable format.
    :param hyperparams_dict: Dictionary of hyperparameter values
    :param fit_params_dict: Dictionary of fit parameter values
    :param best: The best hyperparameters as returned by hyperopt
    :param file_name: Directory plus name of the file you want to save the best parameters to.  File name must end in .json as this is the expected output format
    :return: Parameter dictionary. Contains both model hyperparameters and epochs parameter for model fit.
    """
    # Extract hyperparameters for the model
    best_params = {}
    for key, val in best.items():
        if key in hyperparams_dict:
            input_ = hyperparams_dict[key]
            if input_[0] == 'choice':
                best_params[key] = input_[1][val]
            else:
                best_params[key] = val

    # The only other parameter I need to get out is the number of epochs to train for.
    # I'll put it all into the best_params dictionary, but I'll need to pop it out before defining the model
    best_params['num_epochs'] = fit_params_dict['num_epochs'][1][best['num_epochs']]

    if file_name is not None:
        json_out =json.dumps(best_params)
        f = open(file_name, "w")
        f.write(json_out)
        f.close()

    return best_params


def get_best_hyperparams_dataiku(folder, hyperparams_dict, fit_params_dict, best, file_name=None):
    """
    Helper function to extract the numerical values of best hyperparameters from hyperopt into a more easily usable format within a dataiku environment
    :param folder: The dataiku folder object
    :param hyperparams_dict: Dictionary of hyperparameter values
    :param fit_params_dict: Dictionary of fit parameter values
    :param best: The best hyperparameters as returned by hyperopt
    :param file_name: Directory plus name of the file you want to save the best parameters to.  File name must end in .json as this is the expected output format
    :return: Parameter dictionary. Contains both model hyperparameters and epochs parameter for model fit.
    """
    # Extract hyperparameters for the model
    best_params = {}
    for key, val in best.items():
        if key in hyperparams_dict:
            input_ = hyperparams_dict[key]
            if input_[0] == 'choice':
                best_params[key] = input_[1][val]
            else:
                best_params[key] = val

    # The only other parameter I need to get out is the number of epochs to train for.
    # I'll put it all into the best_params dictionary, but I'll need to pop it out before defining the model
    best_params['num_epochs'] = fit_params_dict['num_epochs'][1][best['num_epochs']]

    if file_name is not None:
        json_out = json.dumps(best_params)
        handle = dataiku.Folder(folder)
        with handle.get_writer(file_name) as w:
            w.write(json_out.encode())

    return best_params


def load_best_params(file_name):
    """
    Function to load previously saved best parameters
    :param file_name: Directory and name of the json file to load
    :return: Dictionary with parameters contained in the input json file
    """
    with open(file_name, 'r') as f:
        best_params = json.load(f)

    return best_params


def load_best_params_dataiku(folder, file_name):
    """
    Function to load previously saved best parameters in a dataiku environment
    :param folder: The dataiku folder object
    :param file_name: Directory and name of the json file to load
    :return: Dictionary with parameters contained in the input json file
    """
    handle = dataiku.Folder(folder)
    path = handle.get_path()
    with open(path + '/' + file_name, 'r') as f:
        best_params = json.load(f)

    return best_params


def fit_eval(params, eval_metric, train_interactions, valid_interactions, num_epochs, num_threads, item_features=None, user_features=None, k=10, verbose=True):
    """
    Helper function to fit LightFM model with desired parameters and automatically compute desired evaluation metric
    :param params: Dictionary of hyperparameters to pass to the LightFM model
    :param eval_metric: The evaluation metric you want to use
    :param train_interactions: The training set of user/item interactions
    :param valid_interactions: The validation set of user/item interactions used to compute the evaluation metric of choice
    :param num_epochs: The number of epochs to train the model
    :param num_threads: The number of threads to use for training
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :param k: The k parameter for precision at k and recall at k metrics.  Optional. Doesn't do anything if you aren't using one of these metrics
    :param verbose: Controls the verbosity of the model fit.  Default is True. This means the model will print out the epoch it's on as it's training
    :return:
    """
    model = LightFM(**params)
    model.fit(interactions=train_interactions,
              epochs=num_epochs,
              num_threads=num_threads,
              item_features=item_features,
              user_features=user_features,
              verbose=verbose)

    if eval_metric == 'auc_score':
        score = auc_score(model,
                          test_interactions=valid_interactions,
                          train_interactions=train_interactions,
                          num_threads=num_threads,
                          item_features=item_features,
                          user_features=user_features).mean()
    elif eval_metric == 'precision_at_k':
        score = precision_at_k(model,
                               test_interactions=valid_interactions,
                               train_interactions=train_interactions,
                               num_threads=num_threads,
                               item_features=item_features,
                               user_features=user_features,
                               k=k).mean()
    elif eval_metric == 'recall_at_k':
        score = recall_at_k(model,
                            test_interactions=valid_interactions,
                            train_interactions=train_interactions,
                            num_threads=num_threads,
                            item_features=item_features,
                            user_features=user_features,
                            k=k).mean()
    elif eval_metric == 'reciprocal_rank':
        score = reciprocal_rank(model,
                                test_interactions=valid_interactions,
                                train_interactions=train_interactions,
                                num_threads=num_threads,
                                item_features=item_features,
                                user_features=user_features).mean()
    else:
        raise ValueError('Invalid evaluation metric provided. Valid choices are auc_score, precision_at_k, recall_at_k, reciprocal_rank')

    return score


def hyperopt_valid(params):
    """
    Fits LightFM model in either cross validation mode or not. Does train/validation split before fitting. Splitting strategy is random, so may result in partial cold start problem upon evaluation.
    :param params: Parameter grid containing hyperparameter search space, parameters to pass to model fit, as well as relevant data, evaluation metric, and evaluation metric parameters if relevant.
                   Should conform to the format expected by hyperopt
    :return: Mean evaluation metric score for all folds per evaluation if running in cross validation mode. Otherwise, it returns the straight mean validation set score (e.g. mean auc score)
    """
    num_epochs = params.pop("num_epochs")
    test_percentage = params.pop('test_percentage')
    all_data = params.pop('data')
    item_features = params.pop('item_features')
    user_features = params.pop('user_features')
    cv = params.pop('cv')
    num_threads = params.pop('num_threads')
    eval_metric = params.pop('eval_metric')
    k = params.pop('k')
    verbose = params.pop('verbose')

    # TODO: build in option to do informed train/valid splitting to avoid partial cold start predictions if desired.
    if cv is not None:
        fold_results_list = []

        for fold in range(cv):
            train_, valid_ = random_train_test_split(interactions=all_data, test_percentage=test_percentage)

            score = fit_eval(params=params,
                             eval_metric=eval_metric,
                             train_interactions=train_,
                             valid_interactions=valid_,
                             num_epochs=num_epochs,
                             num_threads=num_threads,
                             item_features=item_features,
                             user_features=user_features,
                             k=k,
                             verbose=verbose)

            fold_results_list.append(score)

            print('completed fold: {0}'.format(len(fold_results_list)))

        print('completed all folds in current iteration, starting next iteration')

        return np.mean(fold_results_list)
    else:
        train_, valid_ = random_train_test_split(interactions=all_data, test_percentage=test_percentage)

        score = fit_eval(params=params,
                         eval_metric=eval_metric,
                         train_interactions=train_,
                         valid_interactions=valid_,
                         num_epochs=num_epochs,
                         num_threads=num_threads,
                         item_features=item_features,
                         user_features=user_features,
                         k=k,
                         verbose=verbose)

        return score


def f_objective(params):
    """
    Objective function to minimize for hyperopt parameter search.  This is the actual objective function used to measure performance, not the surrogate model or the expected improvement
    :param params: Dictionary with the hyperparameter search space. Expected to be conform to the expected format from hyperopt
    :return: Dictionary entry with the current loss value.  Will be added to the history and used to inform the next round for hyperparameter selection
    """
    loss = hyperopt_valid(params)
    return {'loss': -loss, 'status': STATUS_OK}


def fit_cv(params, interactions, eval_metric, num_epochs, num_threads, test_percentage=None, item_features=None, user_features=None, cv=None, k=10, seed=None, refit=False, verbose=True):
    """
    Fit a LightFM model with or without cross validation. The idea here is to randomly divide the training data and build a model on one set and test on a another to measure performance stability
    :param params: The hyperparameters dictionary to pass to LightFM model
    :param interactions: The full training set of user/item interactions
    :param eval_metric: The evaluation metric you want to use for the cv folds
    :param num_epochs: The number of epochs to run the model for
    :param num_threads: The number of threads to use
    :param test_percentage: The percentage of the input interactions you want to use to create the train/test sets per fold
    :param item_features: The sparse matrix of features for the items
    :param user_features: The sparse matrix of features for the users
    :param cv: The number of cross validation folds
    :param k: The k parameter for the precision at k and recall at k metrics.  Only relevant if you are using one of these metrics for valuation
    :param seed: The random seed to use in model building.  Doesn't apply to the train/test split, but should start the model training at the same place
    :param refit: If true, refit the model to the entire training data once the cv evaluation is complete
    :param verbose: Controls the verbosity of the model fit.  Default is True. This means the model will print out the epoch it's on as it's training
    :return: if refit is True, returns the model trained on the full training set and a list of cross validation scores for each fold.  Otherwise, just the cross validation scores list is returned
    """
    if test_percentage is None:
        raise ValueError('Please provide a test_percentage to split the input training data')

    if seed is not None:
        params['random_state'] = np.random.RandomState(seed)
    else:
        print('The random seed is not set.  This will lead to potentially non-reproducible results.')

    if cv is not None:
        print('Fitting model in cross validation mode for {0} folds'.format(cv))

        # Initialize a list to store cross validation results
        # TODO: Generalize this to multi-evaluation metric outputs
        score_list = []

        for fold in range(cv):

            # Do not set the seed here. It will just result in the same split over and over again in the loop
            # TODO: Allow for different split strategies.  This is totally random and could result in a partial cold start problem.
            train_, valid_ = random_train_test_split(interactions=interactions, test_percentage=test_percentage)

            score = fit_eval(params=params,
                             eval_metric=eval_metric,
                             train_interactions=train_,
                             valid_interactions=valid_,
                             num_epochs=num_epochs,
                             num_threads=num_threads,
                             item_features=item_features,
                             user_features=user_features,
                             k=k,
                             verbose=verbose)

            score_list.append(score)

            print('Completed fold: {0}'.format(fold+1))
            print('Fold {0} loss: {1}'.format(fold+1, score))

        print('Cross validation complete.')

        if refit:
            print('Refitting model to all provided training data')
            model = LightFM(**params)
            model.fit(interactions=interactions,
                      epochs=num_epochs,
                      num_threads=num_threads,
                      item_features=item_features,
                      user_features=user_features,
                      verbose=verbose)

            return model, score_list
        else:
            print('Not refitting model. Returning only cross validation scores')

            return score_list
    else:
        print('Not running in cross validation mode. No model testing will occur, only one fit')

        model = LightFM(**params)
        model.fit(interactions=interactions,
                  epochs=num_epochs,
                  num_threads=num_threads,
                  item_features=item_features,
                  user_features=user_features,
                  verbose=verbose)

        return model


def save_model(model, filename):
    """
    Function to save model
    :param model: The trained model to save
    :param filename: The directory and name of the file you want to use. The format is .npz
    :return: Nothing, just saves the model object
    """
    model_params = {value: getattr(model, value) for value in possible_model_weights}
    hyperparams = model.get_params()
    model_params.update(hyperparams)

    np.savez_compressed(filename, **model_params)

    if os.path.isfile(filename):
        print('Model saved')
    else:
        print('Something went wrong. Model not saved.')


def save_model_dataiku(folder, model, filename):
    """
    Function to save model in a dataiku environment
    :param folder: The dataiku folder object
    :param model: The trained model to save
    :param filename: The directory and name of the file you want to use. The format is .npz
    :return: Nothing, just saves the model object
    """
    model_params = {value: getattr(model, value) for value in possible_model_weights}
    hyperparams = model.get_params()
    model_params.update(hyperparams)

    # np.savez_compressed(filename, **model_params)

    # if os.path.isfile(filename):
    #      print('Model saved')
    #  else:
    #      print('Something went wrong. Model not saved.')

    if filename is not None:
        handle = dataiku.Folder(folder)
        with handle.get_writer(filename) as w:
            # TODO: Again, is there a reason to prefer pickle over numpy sparse matrix saving? I worry about reloading this object.
            pickle.dump(model_params, w, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(filename):
    """
    Function to load saved model
    :param filename: The directory and name of the .npz file with the saved model
    :return: The saved model object
    """
    model = LightFM()

    numpy_model = np.load(filename, allow_pickle=True)
    for value in [x for x in numpy_model if x in possible_model_weights]:
        setattr(model, value, numpy_model[value])

    model.set_params(**{k: v for k, v in numpy_model.items() if k not in possible_model_weights})

    return model


def load_model_dataiku(folder, filename):
    """
    Function to load saved model in a dataiku environment
    :param folder: The dataiku folder object
    :param filename: The directory and name of the .npz file with the saved model
    :return: The saved model object
    """
    model = LightFM()

    handle = dataiku.Folder(folder)
    path = handle.get_path()

    # TODO: Will this work when saving as a pickle and not using the numpy savez_compressed?
    numpy_model = np.load(path + "/" + filename, allow_pickle=True)
    for value in [x for x in numpy_model if x in possible_model_weights]:
        setattr(model, value, numpy_model[value])

    model.set_params(**{k: v for k, v in numpy_model.items() if k not in possible_model_weights})

    return model


def output_checkpoint_files(hyperparams_dict, fit_params_dict, batch_number, trials, best):
    """
    Function to output the trials history and best found hyperparameters so far during the search phase.  Works with the batch search in fit_model_batch.
    :param hyperparams_dict: Dictionary of hyperparameter values
    :param fit_params_dict: Dictionary of fit parameter values
    :param batch_number: The max_evals in this batch.  This is the i index in fit_model_batch.
    :param trials: Trials object from hyperopt fmin
    :param best: The best found parameters dictionary output by hyperopt fmin
    :return: Nothing. Just saves the files with the current time and batch number
    """
    # Get the current time to index the output files.
    # We output the current trials object to restart the search at the appropriate place and the current best parameters found.
    current_time = strftime("%Y-%m-%d_%H%M%S", gmtime())
    file_name_trials = "dump_trials_at_eval_{0}_{1}.p".format(batch_number, current_time)
    file_name_best_raw = "dump_best_params_raw_output_at_eval_{0}_{1}.p".format(batch_number, current_time)
    file_name_best_params_formatted = "dump_best_params_at_eval_{0}_{1}.p".format(batch_number, current_time)

    # Output the trials object.  You need this to restart the search at the end of this batch if you want to.
    pickle.dump(trials, open(file_name_trials, "wb"))

    # Reload the trials object to restart the search.
    # TODO: Do I need to to do here, or can I just re-assign the variable?
    # trials = pickle.load(open(file_name_trials, "rb"))

    # Dump the best seen parameters so far.
    # TODO: format this appropriately
    pickle.dump(best, open(file_name_best_raw, "wb"))

    # Get the formatted hyperparameters and fit parameters found so far
    get_best_hyperparams(hyperparams_dict, fit_params_dict, best, file_name=file_name_best_params_formatted)

    return file_name_trials


def output_checkpoint_files_dataiku(folder, hyperparams_dict, fit_params_dict, batch_number, trials, best):
    """
    Function to output the trials history and best found hyperparameters so far during the search phase.  Works with the batch search in fit_model_batch. Works in dataiku environment
    :param folder: The dataiku folder object
    :param hyperparams_dict: Dictionary of hyperparameter values
    :param fit_params_dict: Dictionary of fit parameter values
    :param batch_number: The max_evals in this batch.  This is the i index in fit_model_batch.
    :param trials: Trials object from hyperopt fmin
    :param best: The best found parameters dictionary output by hyperopt fmin
    :return: Nothing. Just saves the files with the current time and batch number
    """
    # Get the current time to index the output files.
    # We output the current trials object to restart the search at the appropriate place and the current best parameters found.
    current_time = strftime("%Y-%m-%d_%H%M%S", gmtime())
    file_name_trials = "dump_trials_at_eval_{0}_{1}.p".format(batch_number, current_time)
    file_name_best_raw = "dump_best_params_raw_output_at_eval_{0}_{1}.p".format(batch_number, current_time)
    file_name_best_params_formatted = "dump_best_params_at_eval_{0}_{1}.p".format(batch_number, current_time)

    # Setup the needed dataiku handle
    handle = dataiku.Folder(folder)

    # Output the trials object.  You need this to restart the search at the end of this batch if you want to.
    with handle.get_writer(file_name_trials) as w:
        pickle.dump(trials, open(w, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # Reload the trials object to restart the search.
    # TODO: Do I need to to do here, or can I just re-assign the variable?
    # trials = pickle.load(open(file_name_trials, "rb"))

    # Dump the best seen parameters so far.
    # TODO: format this appropriately
    with handle.get_writer(file_name_trials) as w:
        pickle.dump(best, open(w, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # Get the formatted hyperparameters and fit parameters found so far
    get_best_hyperparams_dataiku(folder, hyperparams_dict, fit_params_dict, best, file_name=file_name_best_params_formatted)

    return file_name_trials



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

Version 0.0.1

"""


# Import modules

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle

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


def fit_model_batch(interactions, hyperparams_dict, fit_params_dict, batch_size, test_percentage=0.1, item_features=None, user_features=None, cv=None, random_search=False, hyper_opt_search=True, max_evals=10, seed=0, eval_metric='auc_score', k=10, verbose=True):
    """
    Higher level function to actually run all the aspects of the hyperparameter search.
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
    :param cv: The number of cross validation folds to use.  Should be an interger number of folds or None if you don't want to run with cross validation
    :param random_search: True if you want to use randomized search over the parameters
    :param hyper_opt_search: True if you want to use tree parzen estimators algorithm for parameter search
    :param max_evals: The maximum number of evaluations to use for parameter search
    :param seed: The random seed to use in model building.  Doesn't apply to the train/test split, but should start the model training at the same place
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
        # Iterate over the maximum evaluations by batch size
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
    :param cv: The number of cross validation folds to use.  Should be an interger number of folds or None if you don't want to run with cross validation
    :param random_search: True if you want to use randomized search over the parameters
    :param hyper_opt_search: True if you want to use tree parzen estimators algorithm for parameter search
    :param max_evals: The maximum number of evaluations to use for parameter search
    :param seed: The random seed to use in model building.  Doesn't apply to the train/test split, but should start the model training at the same place
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
    Formats the input range of hyperparameters and fit parameters to search over for model optimization for us in HyperOpt
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


def load_best_params(file_name):
    """
    Function to load previously saved best parameters
    :param file_name: Directory and name of the json file to load
    :return: Dictionary with parameters contained in the input json file
    """
    with open(file_name, 'r') as f:
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
    :return:
    """
    model_params = {value: getattr(model, value) for value in possible_model_weights}
    hyperparams = model.get_params()
    model_params.update(hyperparams)

    np.savez_compressed(filename, **model_params)

    if os.path.isfile(filename):
        print('Model saved')
    else:
        print('Something went wrong. Model not saved.')


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

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


from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll.stochastic
from sklearn.model_selection import cross_val_score


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

sys.path.append('/Users/danielokeeffe/Documents/src/recommender_utils')
import recommender_utils
import sb_utils


def fit_model(interactions, hyperparams_dict, fit_params_dict, test_percentage=0.1, item_features=None, user_features=None, cv=None, random_search=False, hyper_opt_search=True, max_evals=10, seed=0, eval_metric='auc_score', k=10):
    """

    :param interactions:
    :param hyperparams_dict:
    :param fit_params_dict:
    :param test_percentage:
    :param item_features:
    :param user_features:
    :param cv:
    :param random_search:
    :param hyper_opt_search:
    :param max_evals:
    :param seed:
    :return:
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
                                      k=k)
    trials = Trials()

    if random_search:
        best = fmin(f_objective, params, algo=tpe.rand.suggest, max_evals=max_evals, trials=trials, rstate=np.random.RandomState(seed))

        return best, trials

    if hyper_opt_search:
        best = fmin(f_objective, params, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.RandomState(seed))

        return best, trials


def prep_params_for_hyperopt(hyperparams_dict, fit_params_dict, interactions, test_percentage, item_features, user_features, cv, eval_metric, k):
    """

    :param hyperparams_dict:
    :param fit_params_dict:
    :param interactions:
    :param test_percentage:
    :param item_features:
    :param user_features:
    :param cv:
    :param eval_metric:
    :param k:
    :return:
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

    return params


def get_best_hyperparams(hyperparams_dict, fit_params_dict, best):
    """
    Helper function to extract the numerical values of best hyperparameters from hyperopt into a more easily usable format.
    :param hyperparams_dict: Dictionary of hyperparameter values
    :param fit_params_dict: Dictionary of fit parameter values
    :param best: The best hyperparameters as returned by hyperopt
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

    return best_params


def fit_eval(params, eval_metric, train_interactions, valid_interactions, num_epochs, num_threads, item_features=None, user_features=None, k=10):
    """

    :param params:
    :param eval_metric:
    :param train_interactions:
    :param valid_interactions:
    :param num_epochs:
    :param num_threads:
    :param item_features:
    :param user_features:
    :return:
    """
    model = LightFM(**params)
    model.fit(interactions=train_interactions,
              epochs=num_epochs,
              num_threads=num_threads,
              item_features=item_features,
              user_features=user_features)

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

    # TODO: build in option to do informed train/valid splitting to avoid partial cold start predictions if desired.
    if cv:
        print('Running in cross validation mode for {0} folds'.format(cv))
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
                             k=k)

            fold_results_list.append(score)

            print('completed fold: {0}'.format(len(fold_results_list)))

        print('completed all folds in current iteration, starting next iteration')

        return np.mean(fold_results_list)
    else:
        print('Not running in cross validation mode. Will default to single train test split')

        train_, valid_ = random_train_test_split(interactions=all_data, test_percentage=test_percentage)

        score = fit_eval(params=params,
                         eval_metric=eval_metric,
                         train_interactions=train_,
                         valid_interactions=valid_,
                         num_epochs=num_epochs,
                         num_threads=num_threads,
                         item_features=item_features,
                         user_features=user_features,
                         k=k)

        return score


def f_objective(params):
    """
    Objective function to minimize for hyperopt parameter search.  This is the actual objective function used to measure performance, not the surrogate model or the expected improvement
    :param params: Dictionary with the hyperparameter search space. Expected to be conform to the expected format from hyperopt
    :return: Dictionary entry with the current loss value.  Will be added to the history and used to inform the next round for hyperparameter selection
    """
    loss = hyperopt_valid(params)
    return {'loss': -loss, 'status': STATUS_OK}


def fit_cv(params, interactions, eval_metric, num_epochs, num_threads, test_percentage=None, item_features=None, user_features=None, cv=None, k=10, seed=None, refit=False):
    """

    :param params:
    :param interactions:
    :param eval_metric:
    :param num_epochs:
    :param num_threads:
    :param test_percentage:
    :param item_features:
    :param user_features:
    :param cv:
    :param k:
    :param seed:
    :param refit:
    :return:
    """
    if not test_percentage:
        raise ValueError('Please provide a test_percentage to split the input training data')

    if seed:
        params['random_state'] = np.random.RandomState(seed)
    else:
        print('The random seed is not set.  This will lead to potentially non-reproducible results.')

    if cv:
        print('Fitting model in cross validation model for {0} folds'.format(cv))

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
                             k=k)

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
                      user_features=user_features)

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
                  user_features=user_features)

        return model

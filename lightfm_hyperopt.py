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


def fit_model(interactions, hyperparams_dict, fit_params_dict, test_percentage=0.1, item_features=None, user_features=None, cv=None, random_search=False, hyper_opt_search=True, max_evals=10, seed=0, eval_metric='auc', k=10):
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
    :return:
    """
    params = {}

    # Parse the hyperparameters dictionary
    for key, val in hyperparams_dict.items():
        if isinstance(val, list):
            arg_ = [key] + val[1::]
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
            else:
                raise ValueError('Unsupported hyperparameter distribution {0} provided'.format(val[0]))
        else:
            if val[0] == 'randint':
                arg_ = [key] + val[1::]
                params[key] = hp.randint(*arg_)

    # Parse the fit parameters dictionary
    for key, val in fit_params_dict.items():
        # These should only be discrete choice distributions
        arg_ = [key] + val[1::]
        params[key] = hp.choice(*arg_)

    # Parse the interactions data
    params['all_data'] = hp.choice('all_data', [interactions])

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
        raise ValueError('Invalid evaluation metric provided.')

    return score


def hyperopt_valid(params):
    """

    :param params:
    :return:
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

        print('completed fold: {0}'.format(len(fold_results_list)))

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

    :param params:
    :return:
    """
    loss = hyperopt_valid(params)
    return {'loss': -loss, 'status': STATUS_OK}


def extract_best_params_for_refit(hyperparams_dict, fit_params_dict, best):
    """

    :param hyperparams_dict:
    :param fit_params_dict:
    :param best:
    :return:
    """
    pass


# class LightFMHyperOpt(LightFM):
#     """
#     Docs
#     """
#
#     def __init__(self, param_grid, eval_metric, random_search=False, hyper_opt_search=True, max_evals=None):
#         """
#
#         :param param_grid:
#         :param eval_metric:
#         :param random_search:
#         :param hyper_opt_search:
#         :param max_evals:
#         """
#         super().__init__()
#         self.param_grid = param_grid
#         self.eval_metric = eval_metric
#         self.random_search = random_search
#         self.hyper_opt_search = hyper_opt_search
#         self.max_evals = max_evals
#
#         def sample_hyperparameters(param_grid):
#             """
#             Yield possible hyperparameter choices.
#             :return: Nothing
#             """
#
#             while True:
#                 yield param_grid
#
#         def random_search(train, test, param_grid, item_features=None, user_features=None, num_samples=10, num_threads=1):
#             """
#             Run a random search on provided hyperparameter distribution
#             :param train:
#             :param test:
#             :param param_grid:
#             :param item_features:
#             :param user_features:
#             :param num_samples:
#             :param num_threads:
#             :return:
#             """
#             for hyperparams in itertools.islice(sample_hyperparameters(param_grid), num_samples):
#                 num_epochs = hyperparams.pop("num_epochs")
#
#                 model = LightFM(**hyperparams)
#                 model.fit(train, epochs=num_epochs, num_threads=num_threads, item_features=item_features)
#
#                 score = auc_score(model, test, train_interactions=train, num_threads=num_threads, item_features=item_features, user_features=user_features).mean()
#
#                 hyperparams["num_epochs"] = num_epochs
#
#                 yield (score, hyperparams, model)

# def sample_hyperparameters(param_grid):
#     """
#     Yield possible hyperparameter choices.
#     :param param_grid: input grid of hyperparameters to sample from
#     :return: Nothing
#     """
#     while True:
#         yield param_grid
#
#
# def random_search(train):
#     pass

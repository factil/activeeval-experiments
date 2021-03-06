import os
import tqdm
import numpy as np
import pandas as pd

from experiments import (AISHExperiment, AISHShallowExperiment, 
                         PassiveExperiment, AISIExperiment, 
                         StratifiedExperiment, OASISExperiment, ISExperiment, 
                         load_pool, result_to_dataframe, compute_true_measure)

from activeeval.evaluator import Evaluator
from activeeval.proposals import (Passive, StaticVarMin, PartitionedStochasticOE, PartitionedDeterministicOE,
                                  PartitionedAdaptiveVarMin, PartitionedIndepOE,
                                  HierarchicalStochasticOE, HierarchicalDeterministicOE, AdaptiveBaseProposal,
                                  AdaptiveVarMin, compute_optimal_proposal)

from sampling.uniform_stratified import StratifiedUniformSampler
from sampling.sampler import Sampler
from sampling.stratification import Strata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plotting
from plotting import plot_convergence, plot_results

from activeeval.pools import HierarchicalStratifiedPool
from activeeval.measures import FMeasure

init_wd = os.getcwd()

# Run experiments on these datasets
h5_paths = ['datasets/abt-buy-svm.h5',
            'datasets/amazon-googleproducts-svm.h5',
            'datasets/dblp-acm-svm-small.h5',
            'datasets/restaurant-svm.h5',
            'datasets/safedriver-xgb.h5',
            'datasets/creditcard-lr.h5',
            'datasets/tweets100k-svm.h5']

# Specify custom y-axis limits for datasets above (excluding .h5 extension)
mse_ylims = {'safedriver-xgb': (2e-5, 5e-4),
             'tweets100k-svm': (2e-5, 1e-3)}

# Non-default method names to show in plots. It's best if these are short.
map_expt_name = {'AISHExperiment': 'Ours-8',
                 'AISHShallowExperiment': 'Ours-1',
                 'OASISExperiment': 'OASIS',
                 'PassiveExperiment': 'Passive',
                 'ISExperiment': 'IS',
                 'StratifiedExperiment': 'Stratified'}

# Data set names to show in plots.
map_data_name = {'abt-buy-svm': 'abt-buy',
                 'amazon-googleproducts-svm': 'amzn-goog',
                 'dblp-acm-svm-small': 'dblp-acm',
                 'restaurant-svm': 'restaurant',
                 'safedriver-xgb': 'safedriver',
                 'creditcard-lr': 'creditcard',
                 'tweets100k-svm': 'tweets100k'}

n_queries = [50]*100
n_repeats = 1000
n_processes = 20
tree_depth = 8
n_children = 2
max_error = 1
compute_kl_div = True
deterministic = False
em_tol = 1e-6
em_max_iter = 1000
expt_types = [AISHExperiment, AISHShallowExperiment, OASISExperiment]
working_dir = "./results/f1-score"

os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)


def save_hist(sample, true_mean, title, bins=40):
    fig, ax = plt.subplots(1, 1)
    ax.hist(sample, range=(0, 1), bins=bins, density=True)
    ax.axvline(true_mean, color='red')
    ax.set_title(title)
    fig.savefig(f"{title}.png")


if __name__ == '__main__':
    for h5_path in h5_paths:
        # Since we changed directory, use initial working directory as reference
        h5_path = os.path.join(init_wd, h5_path)
        print("Working on experiments for dataset at '{}'".format(h5_path))

        _, true_labels, scores, probs, preds, dataname = load_pool(h5_path)
        measure = FMeasure(preds)
        prior = np.c_[1 - probs, probs]
        true_label_dist = np.c_[1 - true_labels, true_labels]
        labels = np.asarray([0, 1])
        true_measure = compute_true_measure(measure, true_label_dist, labels)
        print(h5_path)
        print("true_measure", true_measure)
        print("len(labels)", len(true_labels))
        result = []
        for _ in tqdm.tqdm(range(200)):
            strata = Strata.from_usm(probs)
            sampler = StratifiedUniformSampler(0.5, probs, strata)
            seen = set()
            n_queries = 5000
            while len(seen) < n_queries:
                idx = sampler.select()
                instance_id = strata.sample_in_strata(idx)
                if instance_id not in seen:
                    seen.add(instance_id)
                sampler.set(instance_id, true_labels[instance_id])

            result.append(sampler.f_score_history()[-1])

            # sampler = Sampler(StratifiedUniformSampler, 0.5, probs, [], [])
            # sampler.sample(lambda x: true_labels[x], 5000)
            # est = sampler.f_score_history()[-1]
            # result.append(est)
            # print(est)
        print(result)
        save_hist(result, true_measure[0], f"{h5_path}-base-um")

        f_scores = []
        for _ in tqdm.tqdm(range(200)):
            try:
                pool = HierarchicalStratifiedPool(scores, tree_depth, n_children,
                                                  bins='csf', bins_hist='fd')
            except ValueError:
                # Sqrt-method for setting the number of bins for the scores histogram
                # has failed (too few bins)
                # Set heuristically to 4 * n_strata
                if np.size(n_children) == 1:
                    bins_hist = 4 * n_children**tree_depth
                else:
                    bins_hist = 4 * np.multiply.reduce(n_children).astype(int)
                pool = HierarchicalStratifiedPool(scores, tree_depth, n_children,
                                                  bins='csf', bins_hist=bins_hist)

            oracle_estimator = HierarchicalDeterministicOE(pool, labels,
                                                           prior=prior,
                                                           em_max_iter=em_max_iter,
                                                           em_tol=em_tol
                                                           )
            proposal = AdaptiveVarMin(pool, measure, oracle_estimator)
            evaluator = Evaluator(pool, measure, proposal)
            n_queries = 5000
            seen = set()
            while len(seen) < n_queries:
                instance_id, weight = evaluator.query()
                if instance_id not in seen:
                    seen.add(instance_id)
                label = true_labels[instance_id]
                evaluator.update(instance_id, label, weight)

            f_scores.append(evaluator.estimate)
            print(evaluator.estimate)
        save_hist(f_scores, true_measure[0], f"{h5_path}-new-det")
        print(f_scores)

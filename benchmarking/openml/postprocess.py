import os
import json
import collections
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import pandas as pd

class PostProcess:

    def __init__(self, results_dir, plots_dir):
        self.results_dir = results_dir
        self.plots_dir = plots_dir

    @property
    def optimizers(self):
        return list(os.walk(self.results_dir))[0][1]

    @property
    def task_results(self):
        results = collections.defaultdict(dict)
        for optimizer in self.optimizers:
            for result_file in list(os.walk(os.path.join(self.results_dir, optimizer)))[0][2]:
                result_file = os.path.join(self.results_dir, optimizer, result_file)
                if not re.match("^.*json$", result_file):
                    continue

                with open(result_file, 'r') as f:
                    result = json.load(f)
                if 1: # result['max_evals'] == 50 and result['batch_size'] == 5:
                    results[result['task_id']][optimizer] = dict(scores=np.array(result['scores']), experiments=result.get('experiments', []))
                else:
                    print("Not valid result %s" % result)

        return results

    def scatter_matrix(self, file_name, search_path):
        plots_dir = self.plots_dir
        sns.set(style="ticks")

        # df = sns.load_dataset("iris")
        # sns.pairplot(df, hue="species")
        df = pd.DataFrame(search_path)
        df = pd.get_dummies(df)
        sns.pairplot(df, hue='score')

        fig_file = os.path.join(plots_dir, file_name + '.svg')
        plt.savefig(fig_file)
        plt.close()

    def plot(self, file_name, scores):
        plots_dir = self.plots_dir
        plt.figure(figsize=(8, 8))
        optimizer_format = {
            'random_serial': dict(color='blue', linestyle=':'),
            'hp_serial': dict(color='green', linestyle='-.'),
            'mango_serial': dict(color='orange', linestyle='-'),
            'mango_parallel': dict(color='red', linestyle='-'),
            'mango_parallel_cluster': dict(color='cyan', linestyle='-'),
        }
        for optimizer, score in scores.items():
            fmt = optimizer_format[optimizer]
            plt.plot(score, label=optimizer, color=fmt['color'], linestyle=fmt['linestyle'])
        plt.legend(loc="lower right", fontsize=16)
        plt.xlabel("Optimizer iterations", fontsize=16)
        plt.ylabel("Score (roc_auc)", fontsize=16)
        fig_file = os.path.join(plots_dir, file_name + '.svg')
        plt.savefig(fig_file, format='svg', dpi=1200)
        plt.close()


if __name__ == "__main__":
    # collect results
    pp = PostProcess('results5', 'plots5')

    optimizers_required = ['random_serial', 'hp_serial', 'mango_serial'] #, 'mango_parallel_cluster'] # 'mango_parallel', 'mango_parallel_cluster'

    clf = 'xgb'
    print(clf)

    task_results = pp.task_results

    # optimizers_pairplot = ['mango_serial']
    # exp_id = 'copula'  # std scaling features fed to gpr with anistropic length scales
    # for task_id, res in task_results.items():
    #     for optimizer in res.keys():
    #         if optimizer not in optimizers_pairplot:
    #             continue
    #         for idx, experiment in enumerate(res[optimizer]['experiments']):
    #             pp.scatter_matrix("%s-%s-%s%s" % (task_id, optimizer, exp_id, idx), experiment)

    normalized_scores = collections.defaultdict(dict)
    for task_id, res in task_results.items():
        if not all(optimizer in res.keys()
                   for optimizer in optimizers_required):
            print("Not all optimizers for %s; %s" % (task_id, res.keys()))
            continue
        max_random = max(res['random_serial']['scores'])
        for optimizer in optimizers_required:
            normalized_scores[task_id][optimizer] = res[optimizer]['scores'] / max_random

    print(len(list(i for i in normalized_scores if re.match("^%s" % clf, i))))

    for task_id, scores in normalized_scores.items():
        pp.plot(task_id, scores)

    mean_scores = {}
    for optimizer in optimizers_required:
        mean_scores[optimizer] = np.array([scores[optimizer] for task_id, scores in normalized_scores.items()
                                           if re.match("^%s.*" % clf, task_id)])
        mean_scores[optimizer] = np.mean(mean_scores[optimizer], axis=0)

    # print(mean_scores)
    pp.plot("mean_scores_%s" % clf, mean_scores)

import os
import json
import collections
import matplotlib.pyplot as plt
import numpy as np
import re


class PostProcess:

    def __init__(self, results_dir):
        self.results_dir = results_dir

    @property
    def optimizers(self):
        return list(os.walk(self.results_dir))[0][1]

    @property
    def task_results(self):
        results = collections.defaultdict(dict)
        for optimizer in self.optimizers:
            for result_file in list(os.walk(os.path.join(self.results_dir, optimizer)))[0][2]:
                result_file = os.path.join(self.results_dir, optimizer, result_file)
                with open(result_file, 'r') as f:
                    result = json.load(f)
                if result['max_evals'] == 50 and result['batch_size'] == 5:
                    results[result['task_id']][optimizer] = np.array(result['scores'])
                else:
                    print("Not valid result %s" % result)

        return results

    def plot(self, file_name, scores, plots_dir):
        plt.figure(figsize=(8, 8))
        optimizer_format = {
            'random_serial': dict(color='blue', linestyle=':'),
            'hp_serial': dict(color='green', linestyle='-.'),
            'mango_serial': dict(color='orange', linestyle='-'),
            'mango_parallel': dict(color='red', linestyle='-'),
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
    pp = PostProcess('results/')
    # for task_id, scores in pp.task_results.items():
    #     pp.plot(task_id, scores, 'plots')
    task_results = pp.task_results
    optimizers_required = ['random_serial', 'hp_serial',
                           'mango_serial', 'mango_parallel']

    normalized_scores = collections.defaultdict(dict)
    for task_id, scores in task_results.items():
        if not all(optimizer in scores.keys()
                   for optimizer in optimizers_required):
            print("Not all optimizers for %s; %s" % (task_id, scores.keys()))
            continue
        max_random = max(scores['random_serial'])
        for optimizer in optimizers_required:
            normalized_scores[task_id][optimizer] = scores[optimizer] / max_random

    clf = 'svm'
    print(clf)
    print(len(list(i for i in normalized_scores if re.match("^%s" % clf, i))))
    # for task_id, scores in normalized_scores.items():
    #     pp.plot(task_id, scores, 'plots_normalized')

    mean_scores = {}
    for optimizer in optimizers_required:
        mean_scores[optimizer] = np.array([scores[optimizer] for task_id, scores in normalized_scores.items()
                                           if re.match("^%s.*" % clf, task_id)])
        mean_scores[optimizer] = np.mean(mean_scores[optimizer], axis=0)

    pp.plot("mean_scores_%s" % clf, mean_scores, 'plots')

import json, os
from numpy.random import seed
from scipy.stats import pearsonr 
from copy import deepcopy

from constants import *
from utils import get_pred
from compute_ic import check_consistency_for_composition


def compute_hierarchy_accs(model, model_results):
    '''
    This function returns a dict mapping the key associated 
    with the top-level question of a hierarchy with the 
    accuracy the given model achieves at the hierarchy.
    '''
    folder = 'balanced_test_hierarchies'
    files = os.listdir(folder)

    hierarchy_acc = {}
    for file in files:
        if file == 'isolated.json':
            continue

        with open(f'{folder}/{file}', 'r') as f:
            data = json.load(f)

        for key, value in data.items():
            total = 0
            correct = 0

            for question, info in value['subquestion'].items():
                if info['type'] in banned_qtypes or info['answer'] is None:
                    continue
                
                total += 1
                if info['answer'] == get_pred(info, model_results):
                    correct += 1

            if total != 0:
                hierarchy_acc[key] = correct/total * 100

    return hierarchy_acc

def get_overall_ic(consistency_compo):
    '''
    Helper function to compute the overall internal
    consistency score a model achieves within a given
    hierarchy.
    '''
    total = 0
    wrong = 0
    for rule, counts in consistency_compo.items():
        total += counts['Total']
        wrong += counts['Wrong']

    score = 'N/A' if total == 0 else (total - wrong) / total
    return score

def compute_hierarchy_ics(model, model_results):
    '''
    This function returns a dict mapping the key associated 
    with the top-level question of a hierarchy with the 
    internal consistency score the given model achieves 
    at the hierarchy.
    '''
    folder = 'balanced_test_hierarchies'
    files = os.listdir(folder)

    hierarchy_ic = {}
    for file in files:
        if file == 'isolated.json':
            continue

        with open(f'{folder}/{file}', 'r') as f:
            data = json.load(f)

        for key, value in data.items():
            consistency_compo = deepcopy(consistency_per_rule)
            consistency_parent = {}

            subquestions = value['subquestion']
            hierarchy = value['hierarchy']

            for q, comp_subqs in hierarchy.items():
                if subquestions[q]['type'] in banned_qtypes:
                    continue
                q_type = collapsed_qtypes[subquestions[q]['type']]
                if q_type not in consistency_parent:
                    consistency_parent[q_type] = deepcopy(consistency_per_rule)

                for composition, subqs in comp_subqs.items():
                    check_consistency_for_composition(subquestions, hierarchy, q, subqs, consistency_compo,
                                                      consistency_parent, model_results, composition, q_type)

            overall_ic = get_overall_ic(consistency_compo)
            if overall_ic != 'N/A':
                hierarchy_ic[key] = overall_ic
                
    return hierarchy_ic

def compute_correlation(hierarchy_acc, hierarchy_ic):
    '''
    Computes and returns the correlation between hierarchy-wide
    accuracy and internal consistency values.
    '''
    accs, scores = [], []
    for key, acc in hierarchy_acc.items():
        if key not in hierarchy_ic:
            continue

        acc *= 100    
        accs.append(acc)
        score = hierarchy_ic[key]
        scores.append(score)

    seed(1)
    corr, _ = pearsonr(scores, accs)
    return corr        

if __name__ == '__main__':
    # TODO: MODEL NAME. Replace the names in this list with the names of the models to evaluate
    models = ['hcrn', 'hme', 'psac']

    for model in models:
        print(f'Computing Pearson correlation coefficient for {model}')
        with open(f'analysis_results/results_{model}.json', 'r') as f:
            model_results = json.load(f)

        hierarchy_acc = compute_hierarchy_accs(model, model_results)
        hierarchy_ic = compute_hierarchy_ics(model, model_results)
        correlation = compute_correlation(hierarchy_acc, hierarchy_ic)
        print(f'{model} has a correlation of: %.3f' % correlation)

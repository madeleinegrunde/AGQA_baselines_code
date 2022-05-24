import json, os
from numpy.random import seed
from scipy.stats import pearsonr 
from copy import deepcopy

from constants import *
from utils import get_pred
from compute_ic import update_dicts


### IC Code ###

def check_consistency_for_composition(subquestions, hierarchy, q, subqs, consistency_compo,
                                      consistency_parent, model_results, composition, q_type):
    '''
    Helper function to determine which consistency check to apply for the given composition
    rule. To avoid double counting failed consistency checks within a single composition while
    computing the IC score associated with a hierarchy, the consistency checks within this function
    treat the various consistency rules for a single composition rule as one rule.
    '''
    if composition in ['Interaction', 'After', 'Before', 'While']:
        check_interaction_after_before_while(subquestions, q, subqs, consistency_compo,
                                             consistency_parent, model_results,
                                             composition, q_type)
    elif composition in ['Between', 'And']:
        check_between_and(subquestions, q, subqs, consistency_compo, consistency_parent,
                          model_results, composition, q_type)
    elif composition == 'Xor':
        check_xor(subquestions, q, subqs, consistency_compo, consistency_parent,
                  model_results, composition, q_type)
    elif composition == 'Equals':
        check_equals(subquestions, hierarchy, q, subqs, consistency_compo, consistency_parent,
                     model_results, composition, q_type)
    elif composition == 'Choose':
        check_choose(subquestions, hierarchy, q, subqs, consistency_compo, consistency_parent,
                     model_results, composition, q_type)

def check_choose(subquestions, hierarchy, q, subqs, consistency_compo, consistency_parent,
                 model_results, composition, q_type):
    '''
    This function performs consistency checks for the Choose composition rule. There are two
    classes of consistency checks: one where the parent is assumed to be 'before' or 'after'
    and another where the parent is assumed to be an arbitrary open answer.
    '''
    subq1, subq2 = subqs
    subq1_type = subquestions[subq1]['type']
    subq2_type = subquestions[subq2]['type']
    if subq1_type in banned_qtypes or subq2_type in banned_qtypes:
        return

    q_answer = get_pred(subquestions[q], model_results)
    subq1_answer = get_pred(subquestions[subq1], model_results)
    subq2_answer = get_pred(subquestions[subq2], model_results)

    if q_answer == 'before':
        wrong_condition = subq1_answer != 'yes' or subq2_answer != 'no'
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, 'Temporal', wrong_condition)
    elif q_answer == 'after':
        wrong_condition = subq1_answer != 'no' or subq2_answer != 'yes'
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, 'Temporal', wrong_condition)
    else:
        try:
            subq11, _ = hierarchy[subq1]["Equals"]
            subq21, _ = hierarchy[subq2]["Equals"]
        except:
            return
        
        if f'{q_answer} exist' in subq11:
            wrong_condition = subq1_answer != 'yes' or subq2_answer != 'no'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'Object', wrong_condition)
        elif f'{q_answer} exist' in subq21:
            wrong_condition = subq1_answer != 'no' or subq2_answer != 'yes'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'Object', wrong_condition)            

def check_equals(subquestions, hierarchy, q, subqs, consistency_compo, consistency_parent,
                 model_results, composition, q_type):
    '''
    This function performs consistency checks for the Equals composition rule. While there
    are more than two distinct consistency checks for this rule, they are grouped by whether
    the parent is assumed or implied to be 'Yes' or 'No'.
    '''
    if subquestions[q]['type'] != 'Object Equals':
        return

    subq1, subq2 = subqs
    subq1_type = subquestions[subq1]['type']
    subq2_type = subquestions[subq2]['type']
    if subq1_type in banned_qtypes or subq2_type in banned_qtypes:
        return

    q_answer = get_pred(subquestions[q], model_results)
    subq1_answer = get_pred(subquestions[subq1], model_results)
    subq2_answer = get_pred(subquestions[subq2], model_results)

    if subq1 in hierarchy:
        if q_answer == 'yes':
            wrong_condition = subq1_answer != subq2_answer
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'Yes', wrong_condition)
        elif q_answer == 'no':
            wrong_condition = subq1_answer == subq2_answer
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'No', wrong_condition)
        elif subq1_answer == subq2_answer:
            wrong_condition = q_answer != 'yes'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'Yes', wrong_condition)
        elif subq1_answer != subq2_answer:
            wrong_condition = q_answer != 'no'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'No', wrong_condition)
    else:
        subq1_program = subquestions[subq1]['program']
        object = subq1_program[10:-1]
        if q_answer == 'yes':
            wrong_condition = subq2_answer != object or subq1_answer != 'yes'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'Yes', wrong_condition)
        elif q_answer == 'no':
            wrong_condition = subq2_answer == object
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'No', wrong_condition)
        elif subq2_answer != object or subq1_answer != 'yes':
            wrong_condition = q_answer != 'no'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'No', wrong_condition)
        elif subq2_answer == object:
            wrong_condition = q_answer != 'yes'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'Yes', wrong_condition)            

def check_xor(subquestions, q, subqs, consistency_compo, consistency_parent,
              model_results, composition, q_type):
    '''
    This function performs consistency checks for the Xor composition rule. There
    are two consistency checks, one where the parent is assumed or implied to be
    'Yes' and another where the parent is assumed or implied to be 'No'.
    '''
    subq1, subq2 = subqs
    subq1_type = subquestions[subq1]['type']
    subq2_type = subquestions[subq2]['type']
    if subq1_type in banned_qtypes or subq2_type in banned_qtypes:
        return

    q_answer = get_pred(subquestions[q], model_results)
    subq1_answer = get_pred(subquestions[subq1], model_results)
    subq2_answer = get_pred(subquestions[subq2], model_results)
    
    if q_answer == 'yes':
        wrong_condition = subq1_answer != 'yes' or subq2_answer != 'no'
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, 'Yes', wrong_condition)
    elif q_answer == 'no':
        wrong_condition = subq1_answer == 'yes' and subq2_answer == 'no'
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, 'No', wrong_condition)
    elif subq1_answer != 'yes' or subq2_answer != 'no':
        wrong_condition = q_answer != 'no'
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, 'No', wrong_condition)
    elif subq1_answer == 'yes' and subq2_answer == 'no':
        wrong_condition = q_answer != 'yes'
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, 'Yes', wrong_condition)        

def check_between_and(subquestions, q, subqs, consistency_compo, consistency_parent,
                      model_results, composition, q_type):
    '''
    This function performs consistency checks for the And and Between composition rules,
    which follow a similar logic. There are two consistency checks, one where the parent
    is assumed or implied to be 'Yes' and another where the parent is assumed or implied
    to be 'No'.
    '''
    q_answer = get_pred(subquestions[q], model_results)
    if q_answer == 'yes':
        only_yes = True
        one_valid = False
        for subq in subqs:
            subq_type = subquestions[subq]['type']
            if subq_type in banned_qtypes or subq_type not in yesNo:
                continue
            
            subq_answer = get_pred(subquestions[subq], model_results)
            one_valid = True
            if subq_answer != 'yes':
                only_yes = False
        if one_valid:
            wrong_condition = not only_yes
            update_dicts(consistency_compo, consistency_parent,
                         composition, q_type, 'Yes', wrong_condition)
    elif q_answer == 'no':
        oneNo = False
        one_valid = False
        for subq in subqs:
            subq_type = subquestions[subq]['type']
            if subq_type in banned_qtypes or subq_type not in yesNo:
                continue
            
            subq_answer = get_pred(subquestions[subq], model_results)
            one_valid = True
            if subq_answer == 'no':
                oneNo = True
        
        if one_valid:
            wrong_condition = not oneNo
            update_dicts(consistency_compo, consistency_parent,
                         composition, q_type, 'No', wrong_condition)
    else:
        only_yes = True
        oneNo = False
        one_valid = False
        for subq in subqs:
            subq_type = subquestions[subq]['type']
            if subq_type in banned_qtypes or subq_type not in yesNo:
                continue

            one_valid = True
            subq_answer = get_pred(subquestions[subq], model_results)
            if subq_answer != 'yes':
                only_yes = False
            if subq_answer == 'no':
                oneNo = True

        if one_valid:
            if only_yes:
                wrong_condition = q_answer != 'yes'
                update_dicts(consistency_compo, consistency_parent,
                             composition, q_type, 'Yes', wrong_condition)
            if oneNo:
                wrong_condition = q_answer != 'no'
                update_dicts(consistency_compo, consistency_parent,
                             composition, q_type, 'No', wrong_condition)

def check_interaction_after_before_while(subquestions, q, subqs, consistency_compo,
                                         consistency_parent, model_results,
                                         composition, q_type):
    '''
    This function performs consistency checks for the Interaction, After, Before and While
    composition rules. There are two consistency checks, one where the parent is assumed
    or implied to be 'Yes' for the check and another where the parent is assumed or 
    implied to be 'No'.
    '''
    q_answer = get_pred(subquestions[q], model_results)
    if q_answer == 'yes':
        only_yes = True
        one_valid = False

        for subq in subqs:
            subq_type = subquestions[subq]['type']
            if subq_type in banned_qtypes or subq_type not in yesNo:
                continue

            subq_answer = get_pred(subquestions[subq], model_results)
            one_valid = True
            if subq_answer != 'yes':
                only_yes = False

        if one_valid:
            wrong_condition = not only_yes
            update_dicts(consistency_compo, consistency_parent,
                         composition, q_type, 'Yes', wrong_condition)
    else:
         # Contrapositive: Check if there's a child answered 'No'
        oneNo = False
        for subq in subqs:
            subq_type = subquestions[subq]['type']
            if subq_type in banned_qtypes or subq_type not in yesNo:
                continue

            subq_answer = get_pred(subquestions[subq], model_results)
            if subq_answer == 'no':
                oneNo = True

        if oneNo:
            wrong_condition = q_answer != 'no'
            update_dicts(consistency_compo, consistency_parent,
                         composition, q_type, 'No', wrong_condition)


### Correlation code ###

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

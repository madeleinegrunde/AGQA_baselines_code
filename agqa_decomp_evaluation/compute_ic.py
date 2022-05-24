import json, os, pickle
import pandas as pd
import numpy as np
from copy import deepcopy

from utils import mkdir, get_pred
from constants import *

def update_dicts(consistency_compo, consistency_parent,
                 composition, q_type, suffix, wrong_condition):
    '''
    Helper function to update the consistency_compo and consistency_parent
    dicts for the specified consistency rule.
    '''
    consistency_compo[composition + ' ' + suffix]['Total'] += 1
    consistency_parent[q_type][composition + ' ' + suffix]['Total'] += 1
    if wrong_condition:
        consistency_compo[composition + ' ' + suffix]['Wrong'] += 1
        consistency_parent[q_type][composition + ' ' + suffix]['Wrong'] += 1

def check_interaction_after_before_while(subquestions, q, subqs, consistency_compo,
                                         consistency_parent, model_results,
                                         composition, q_type):
    '''
    This function performs consistency checks for the Interaction, After, Before and While
    composition rules. There are two consistency checks, one where the parent is assumed
    or implied to be 'yes' for the check and another where the parent is assumed or 
    implied to be 'no'.
    '''
    q_answer = get_pred(subquestions[q], model_results)

    # "Yes" check
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

    # Contrapositive: Check if there's a child answered 'no'
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

def check_between_and(subquestions, q, subqs, consistency_compo, consistency_parent,
                      model_results, composition, q_type):
    '''
    This function performs consistency checks for the And and Between composition rules,
    which follow a similar logic. There are two consistency checks, one where the parent
    is assumed or implied to be 'yes' and another where the parent is assumed or implied
    to be 'no'.
    '''
    q_answer = get_pred(subquestions[q], model_results)

    # "Yes" checks
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
        only_yes = True
        one_valid = False
        for subq in subqs:
            subq_type = subquestions[subq]['type']
            if subq_type in banned_qtypes or subq_type not in yesNo:
                continue

            one_valid = True
            subq_answer = get_pred(subquestions[subq], model_results)
            if subq_answer != 'yes':
                only_yes = False
        
        if one_valid:
            if only_yes:
                wrong_condition = q_answer != 'yes'
                update_dicts(consistency_compo, consistency_parent,
                             composition, q_type, 'Yes', wrong_condition)
            

    # "No checks"
    if q_answer == 'no':
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
        oneNo = False
        one_valid = False
        for subq in subqs:
            subq_type = subquestions[subq]['type']
            if subq_type in banned_qtypes or subq_type not in yesNo:
                continue

            one_valid = True
            subq_answer = get_pred(subquestions[subq], model_results)
            if subq_answer == 'no':
                oneNo = True

        if one_valid:
            if oneNo:
                wrong_condition = q_answer != 'no'
                update_dicts(consistency_compo, consistency_parent,
                             composition, q_type, 'No', wrong_condition)


def check_xor(subquestions, q, subqs, consistency_compo, consistency_parent,
              model_results, composition, q_type):
    '''
    This function performs consistency checks for the Xor composition rule. There
    are two consistency checks, one where the parent is assumed or implied to be
    'yes' and another where the parent is assumed or implied to be 'no'.
    '''
    subq1, subq2 = subqs
    subq1_type = subquestions[subq1]['type']
    subq2_type = subquestions[subq2]['type']
    if subq1_type in banned_qtypes or subq2_type in banned_qtypes:
        return

    q_answer = get_pred(subquestions[q], model_results)
    subq1_answer = get_pred(subquestions[subq1], model_results)
    subq2_answer = get_pred(subquestions[subq2], model_results)
    
    # "Yes" checks
    if q_answer == 'yes':
        wrong_condition = subq1_answer != 'yes' or subq2_answer != 'no'
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, 'Yes', wrong_condition)
    elif subq1_answer == 'yes' and subq2_answer == 'no':
        wrong_condition = q_answer != 'yes'
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, 'Yes', wrong_condition)        


    # "No" checks
    if q_answer == 'no':
        wrong_condition = subq1_answer == 'yes' and subq2_answer == 'no'
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, 'No', wrong_condition)
    elif subq1_answer != 'yes' or subq2_answer != 'no':
        wrong_condition = q_answer != 'no'
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, 'No', wrong_condition)

def check_equals(subquestions, hierarchy, q, subqs, consistency_compo, consistency_parent,
                 model_results, composition, q_type):
    '''
    This function performs consistency checks for the Equals composition rule. While there
    are more than two distinct consistency checks for this rule, they are grouped by whether
    the parent is assumed or implied to be 'yes' or 'no'.
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
        # "Yes" checks
        if q_answer == 'yes':
            wrong_condition = subq1_answer != subq2_answer
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'Yes', wrong_condition)
        elif subq1_answer == subq2_answer:
            wrong_condition = q_answer != 'yes'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'Yes', wrong_condition)


        # "No" checks
        if q_answer == 'no':
            wrong_condition = subq1_answer == subq2_answer
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'No', wrong_condition)
        elif subq1_answer != subq2_answer:
            wrong_condition = q_answer != 'no'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'No', wrong_condition)
    else:
        subq1_program = subquestions[subq1]['program']
        object = subq1_program[10:-1]

        # "Yes" checks
        if q_answer == 'yes':
            wrong_condition = subq2_answer != object or subq1_answer != 'yes'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'Yes', wrong_condition)
        elif subq2_answer == object:
            wrong_condition = q_answer != 'yes'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'Yes', wrong_condition)            


        # "No" checks
        if q_answer == 'no':
            wrong_condition = subq2_answer == object
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'No', wrong_condition)
        elif subq2_answer != object or subq1_answer != 'yes':
            wrong_condition = q_answer != 'no'
            update_dicts(consistency_compo, consistency_parent, composition,
                         q_type, 'No', wrong_condition)
                    
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

def check_first_last(subquestions, q, subqs, consistency_compo,
                     consistency_parent, model_results, composition, q_type):
    '''
    This function performs consistency checks for the First and Last composition
    rules. The child question for this composition rule always belongs to a banned
    question type. As a result, the check is never applied and is not intended to be
    used.
    '''
    subq = subqs[0]
    if subquestions[subq]['type'] in banned_qtypes:
        return

    q_answer = get_pred(subquestions[q], model_results)
    subq_answer = get_pred(subquestions[subq], model_results)

    if isinstance(subq_answer, list):
        wrong_condition = q_answer not in subq_answer
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, '', wrong_condition)
    else:
        wrong_condition = q_answer != subq_answer
        update_dicts(consistency_compo, consistency_parent, composition,
                     q_type, '', wrong_condition)        

def save_ic_per_check(consistency_compo, model):
    '''
    Computes and saves the internal consistency score for each individual 
    logical consistency rule. The saved csv file contains three columns:
    one for the name of the consistency rule, one for the consistency score
    and another for sample size used to compute the consistency score.
    '''
    # Compute and save scores
    rules = []
    totals = []
    scores = []
    for rule, counts in consistency_compo.items():
        total = counts['Total']
        correct = counts['Total'] - counts['Wrong']
        if total == 0:
            counts['Score'] = 'N/A'
            scores.append(counts['Score'])
        else:
            counts['Score'] = correct / total
            scores.append('%.2f' % (counts['Score'] * 100))
    
        rules.append(rule)
        totals.append(total)

    df = pd.DataFrame.from_dict({'Consistency Rule':rules, f'IC ({model})': scores, f'Total ({model})':totals})
    df.to_csv(f'analysis_results/ic/per_rule_{model}.csv')

def save_ic_per_compo(consistency_compo, model):
    '''
    Computes and saves IC  scores for each composition rule. The saved csv 
    file contains two columns: one for the name of the composition rule
    and another for the IC score.
    '''
    scores = []
    for composition in comp_type_ordering:
        comp_scores = []
        for rule, counts in consistency_compo.items():
            if composition in rule or composition == 'Overall':
                comp_scores.append(counts['Score'])                
        
        score = 'N/A' if 'N/A' in comp_scores or len(comp_scores) == 0 else '%.2f' % (np.mean(comp_scores) * 100)
        scores.append(score)

    df = pd.DataFrame.from_dict({'Composition Rule': comp_type_ordering, f'IC ({model})': scores})
    df.to_csv(f'analysis_results/ic/per_composition_{model}.csv')

def save_ic_per_parent(consistency_parent, model):
    '''
    Computes and saves IC scores for each parent question type. The saved csv 
    file contains two columns: one for the name of the parent question type
    and another for the IC score.
    '''
    scores = []
    for parent_type in subq_type_ordering:
        type_scores = []
        if parent_type not in consistency_parent or parent_type not in parent_to_rules:
            type_scores.append('N/A')
        else:
            count_dict = consistency_parent[parent_type]
            for composition_rule in parent_to_rules[parent_type]:
                for rule, counts in count_dict.items():
                    if composition_rule in rule:
                        total = counts['Total']
                        correct = counts['Total'] - counts['Wrong']
                        score = 'N/A' if total == 0 else correct/total
                        type_scores.append(score)
        
        score = 'N/A' if 'N/A' in type_scores or len(type_scores) == 0 else '%.2f' % (np.mean(type_scores) * 100)
        scores.append(score)
        
    df = pd.DataFrame.from_dict({'Question Type': subq_type_ordering, f'IC ({model})': scores})
    df.to_csv(f'analysis_results/ic/per_parent_type_{model}.csv')

def check_consistency_for_composition(subquestions, hierarchy, q, subqs, consistency_compo,
                                      consistency_parent, model_results, composition, q_type):
    '''
    Helper function to determine which consistency check to apply for the given composition
    rule.
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
    

def check_consistency_for_hierarchy(subquestions, hierarchy, seen_q, consistency_compo,
                                    consistency_parent, model_results):
    '''
    Updates the consistency_compo and consistency_parent dicts to reflect the model's
    performance on the consistency checks applied in the given hierarchy.
    '''
    # Iterate over each composition
    for q, comp_subqs in hierarchy.items():
        # Avoid banned parent questions or double counting
        q_type = subquestions[q]['type']
        if q_type in banned_qtypes or q in seen_q:
            continue
        seen_q.add(q)

        q_type = collapsed_qtypes[q_type]
        if q_type not in consistency_parent:
            consistency_parent[q_type] = deepcopy(consistency_per_rule)

        # Iterate over each composition rule
        for composition, subqs in comp_subqs.items():
            check_consistency_for_composition(subquestions, hierarchy, q, subqs, consistency_compo,
                                              consistency_parent, model_results, composition, q_type)


def compute_ic(model):
    '''
    Main function involved in the computation of the IC metric.
    '''
    consistency_compo = deepcopy(consistency_per_rule)
    consistency_parent = {}
    with open(f'analysis_results/results_{model}.json', 'r') as f:
        model_results = json.load(f)

    folder = 'balanced_test_hierarchies'
    files = os.listdir(folder)
    for file in files:
        if file == 'isolated.json':
            continue

        with open(f'{folder}/{file}', 'r') as f:
            data = json.load(f)
        seen_q = set()

        for key, value in data.items():
            subquestions = value['subquestion']
            hierarchy = value['hierarchy']
            check_consistency_for_hierarchy(subquestions, hierarchy, seen_q,
                                            consistency_compo, consistency_parent, model_results)

    # Report results
    mkdir('analysis_results/ic')
    save_ic_per_check(consistency_compo, model)
    save_ic_per_compo(consistency_compo, model)
    save_ic_per_parent(consistency_parent, model)


if __name__ == '__main__':
    # TODO: MODEL NAME. Replace the names in this list with the names of the models to evaluate
    models = ['hcrn', 'hme', 'psac']

    for model in models:
        print(f'Computing IC scores for {model}')
        compute_ic(model)

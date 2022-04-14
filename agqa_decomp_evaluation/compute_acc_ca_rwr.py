import numpy as np
import json, os, pickle
import pandas as pd
from collections import Counter
import argparse

from constants import *
from utils import mkdir, get_pred


def initialize_count_dict(count_dict, child_keys, parent_keys):
    '''
    Helper function to initialize total and correct question counts
    for a given slice of the dataset. 
    '''
    for p_key in parent_keys:
        if p_key not in count_dict:
            count_dict[p_key] = {}

        for child_key in child_keys:
            if child_key not in count_dict[p_key]:
                count_dict[p_key][child_key] = {'total' : 0, 'correct' : 0}

def compute_isolated_acc(data, acc_count, model_results):
    '''
    Records model accuracy on isolated test questions not associated
    with any hierarchies. 
    '''

    for datapoint in data:
        if datapoint['type'] in banned_qtypes:
            continue

        # Extract answer and preds
        answer = datapoint['answer']
        prediction = get_pred(datapoint, model_results)

        # Initialize dict if not initialized
        subq_type = datapoint['type']
        subq_type = subq_type if subq_type not in collapsed_qtypes else collapsed_qtypes[subq_type] 
        initialize_count_dict(acc_count, [answer], ['Overall', subq_type])

        # Update dict
        correct = 1 if answer == prediction else 0
        for qtype in ['Overall', subq_type]:
            acc_count[qtype][answer]['total'] += 1
            acc_count[qtype][answer]['correct'] += correct                

def compute_acc(subquestions, seen_q_acc, acc_count, model_results):
    '''
    Records model accuracy on subquestions contained in the given hierarchy.
    '''

    for subquestion, info in subquestions.items():
        # Skipped banned question types and questions without answers
        if info['type'] in banned_qtypes or info['answer'] is None:
            continue

        # Avoid double counting
        if subquestion in seen_q_acc:
            continue
        seen_q_acc.add(subquestion)

        answer = info['answer']
        prediction = get_pred(info, model_results)

        subq_type = collapsed_qtypes[info['type']]
        initialize_count_dict(acc_count, [answer], ['Overall', subq_type])

        correct = 1 if answer == prediction else 0
        for qtype in ['Overall', subq_type]:
            acc_count[qtype][answer]['total'] += 1
            acc_count[qtype][answer]['correct'] += correct                        

def compute_ca_rwr(subquestions, hierarchy, seen_q, composition_rule_count,
                   parent_type_count, model_results):
    '''
    Records model performance on the CA, RWR and RWR-n metrics for the given hierarchy.
    '''

    for q, comp_subqs in hierarchy.items():
        info = subquestions[q]
        if info['type'] in banned_qtypes or info['answer'] is None:
            continue

        # Avoid double counting
        if q in seen_q:
            continue
        seen_q.add(q)

        compute_ca_rwr_parent(subquestions, info, hierarchy, comp_subqs, parent_type_count, model_results)
        compute_ca_rwr_composition(subquestions, info, hierarchy, comp_subqs, composition_rule_count,
                                   model_results)

def compute_ca_rwr_parent(subquestions, info, hierarchy, comp_subqs, parent_type_count, model_results):    
    '''
    Records model performance on the CA, RWR and RWR-n metrics while conditioning on the parent question type
    for the given composition.
    '''

    answer = info['answer']
    prediction = get_pred(info, model_results)

    # Get all child questions
    child_questions = set()
    for composition, subqs in comp_subqs.items():
        for subq in subqs:
            child_questions.add(subq)

    # Get number of correctly answered questions
    total = 0
    correct = 0
    for subq in child_questions:
        subq_info = subquestions[subq]
        if subq_info['type'] in banned_qtypes or subq_info['answer'] == None:
            continue

        total += 1
        if subq_info['answer'] == get_pred(subq_info, model_results):
            correct += 1

    # Record results
    if total != 0:
        count_keys = ['ca' if correct == total else 'rwr']
        if correct != total:
            count_keys.append(f'rwr-{total - correct}')

        parent_type = collapsed_qtypes[info['type']]
        initialize_count_dict(parent_type_count, [parent_type, 'Overall'], count_keys)
        
        correct = 1 if answer == prediction else 0
        for count_key in count_keys:
            for qtype in [parent_type, 'Overall']:
                parent_type_count[count_key][qtype]['total'] += 1
                parent_type_count[count_key][qtype]['correct'] += correct

def compute_ca_rwr_composition(subquestions, info, hierarchy, comp_subqs, composition_rule_count,
                               model_results):
    '''
    Records model performance on the CA, RWR and RWR-n metrics while conditioning on composition rule
    for the given composition.
    '''

    answer = info['answer']
    prediction = get_pred(info, model_results)

    for composition, subqs in comp_subqs.items():
        # Get the number of correctly answered children
        total = 0
        correct = 0
        for subq in subqs:
            subq_info = subquestions[subq]
            if subq_info['type'] in banned_qtypes or subq_info['answer'] is None:
                continue

            total += 1
            if subq_info['answer'] == get_pred(subq_info, model_results):
                correct += 1
        
        # Update the dictionary
        if total != 0:
            count_keys = ['ca' if correct == total else 'rwr']
            if correct != total:
                count_keys.append(f'rwr-{total - correct}')

            initialize_count_dict(composition_rule_count, [composition, 'Overall'], count_keys)

            correct = 1 if answer == prediction else 0
            for count_key in count_keys:
                for comp_rule in [composition, 'Overall']:
                    composition_rule_count[count_key][comp_rule]['total'] += 1
                    composition_rule_count[count_key][comp_rule]['correct'] += correct

def save_acc(acc_count, model):
    '''
    Function to save model performance on the accuracy metric. Saves normalized accuracy
    per question type as well as raw counts per question type/ground-truth answer pair. 
    '''

    mkdir('analysis_results/per_gt_acc')

    # Lists holding accuracies
    qtype_gt = []
    total = []
    correct = []
    subquestion_type_accuracy = {}
    for subq_type, answer_dict in acc_count.items():
        answer_accuracies = []
        for answer, count in answer_dict.items():
            qtype_gt.append(f'{subq_type}-{answer}')
            total.append(count['total'])
            correct.append(count['correct'])
            answer_accuracies.append(count['correct']/count['total'])
        subquestion_type_accuracy[subq_type] = np.mean(answer_accuracies)

    # Save sample sizes and raw counts
    df = pd.DataFrame.from_dict({'Type-Gt' : qtype_gt, 'Total' : total, f'Correct ({model})' : correct})
    df.to_csv(f'analysis_results/per_gt_acc/raw_counts_{model}.csv')
    
    # Save normalized accuracies
    accuracies = []
    for subq_type in subq_type_ordering:
        if subq_type not in subquestion_type_accuracy:
            accuracies.append('N/A')
        else:
            accuracies.append('%.2f' % (subquestion_type_accuracy[subq_type] * 100))
    df = pd.DataFrame.from_dict({'Question Type':subq_type_ordering, f'Accuracy ({model})':accuracies})
    df.to_csv(f'analysis_results/per_gt_acc/normalized_acc_{model}.csv')

def save_ca_rwr(parent_type_count, model, parent=True):
    '''
    Function to save model performance on the CA, RWR and RWR-n metrics. For CA and RWR,
    the function saves a csv containing scores on the metrics paired with the total number
    of compositions considered for each question type/composition rule. For RWR-n metrics,
    the function saves scores for all RWR-n variants and raw counts separately.
    '''

    save_ca_rwr_basic(parent_type_count, model, parent)
    save_ca_rwr_granular(parent_type_count, model, parent)

def save_ca_rwr_basic(parent_type_count, model, parent):
    '''
    Function to save model performance on the CA and RWR metrics, as well as the
    Delta metric (RWR - CA).
    '''
    ordering_list = subq_type_ordering if parent else comp_type_ordering
    type_name = 'Question Type' if parent else 'Composition Rule'
    suffix = '_by_parent' if parent else '_by_composition'
    delta_pairs = [[], []]

    for i, count_type in enumerate(['ca', 'rwr']):
        mkdir(f'analysis_results/{count_type}')

        values = []
        counts = []
        for subq_type in ordering_list:
            if subq_type not in parent_type_count[count_type]:
                values.append('N/A')
                delta_pairs[i].append('N/A')
                counts.append(0)
            else:
                type_to_counts = parent_type_count[count_type]
                value = type_to_counts[subq_type]['correct']/type_to_counts[subq_type]['total']
                delta_pairs[i].append(value)
                values.append('%.2f' % (value * 100))
                counts.append(type_to_counts[subq_type]['total'])

        df = pd.DataFrame.from_dict({type_name:ordering_list, f'{count_type.upper()} ({model})':values,
                                     f"Total ({model})":counts})
        df.to_csv(f'analysis_results/{count_type}/{count_type}_{model}{suffix}.csv')

    # Compute Deltas (RWR - CA)
    values = []
    for i in range(len(ordering_list)):
        rwr_i = delta_pairs[1][i]
        ca_i = delta_pairs[0][i]
        value = 'N/A' if 'N/A' in [ca_i, rwr_i] else '%.2f' % ((rwr_i - ca_i) * 100)
        values.append(value)

    mkdir('analysis_results/delta')
    df = pd.DataFrame.from_dict({type_name:ordering_list, f'Delta ({model})':values})
    df.to_csv(f'analysis_results/delta/delta_{model}{suffix}.csv')

def save_ca_rwr_granular(parent_type_count, model, parent):
    '''
    Function to save model performance on the granular RWR-n metrics.
    '''
    ordering_list = subq_type_ordering if parent else comp_type_ordering
    type_name = 'Question Type' if parent else 'Composition Rule'
    suffix = '_by_parent' if parent else '_by_composition'

    rwr_values = {type_name : ordering_list}
    rwr_counts = {type_name : ordering_list}

    for count_type, type_to_counts in parent_type_count.items():
        if count_type in ['ca', 'rwr']:
            continue

        values = []
        counts = []
        for subq_type in ordering_list:
            if subq_type not in type_to_counts:
                values.append('N/A')
                counts.append(0)
            else:
                value = type_to_counts[subq_type]['correct']/type_to_counts[subq_type]['total']
                values.append('%.2f' % (value * 100))
                counts.append(type_to_counts[subq_type]['total'])

        rwr_values[f'{count_type.upper()} ({model})'] = values
        rwr_counts[f'{count_type.upper()} ({model})'] = counts

    pd.DataFrame.from_dict(rwr_values).to_csv(f'analysis_results/rwr/rwr_n_values_{model}{suffix}.csv')
    pd.DataFrame.from_dict(rwr_counts).to_csv(f'analysis_results/rwr/rwr_n_counts_{model}{suffix}.csv')    

def compute_acc_ca_rwr(model):
    '''
    Function to compute and save model performances on the Accuracy, CA, RWR and RWR-n metrics.
    '''

    acc_count = {}
    composition_rule_count = {}
    parent_type_count = {}

    with open(f'analysis_results/results_{model}.json', 'r') as f:
        model_results = json.load(f)

    folder = 'balanced_test_hierarchies'
    files = os.listdir(folder)
    for file in files:
        with open(f'{folder}/{file}', 'r') as f:
            data = json.load(f)

        seen_q_acc = set()
        seen_q_ca_rwr = set()

        # Edge case: isolated questions
        if file == 'isolated.json':
            compute_isolated_acc(data, acc_count, model_results)
            continue

        for key, value in data.items():
            subquestions = value['subquestion']
            hierarchy = value['hierarchy']

            # Per ground-truth accuracy
            compute_acc(subquestions, seen_q_acc, acc_count, model_results)
            
            # Compositional accuracy
            compute_ca_rwr(subquestions, hierarchy, seen_q_ca_rwr, composition_rule_count,
                           parent_type_count, model_results)

    save_acc(acc_count, model)
    save_ca_rwr(parent_type_count, model, parent=True)
    save_ca_rwr(composition_rule_count, model, parent=False)
    
if __name__ == '__main__':
    # TODO: MODEL NAME. Replace the names in this list with the names of the models to evaluate
    models = ['hcrn']

    for model in models:
        print(f'Compute CA, RWR and RWR-n for {model}')
        compute_acc_ca_rwr(model)

import json, os

from constants import banned_qtypes, collapsed_qtypes
from utils import get_pred, mkdir

def split_dataset(model, split_type, only_incorrect):
    # Get folder
    folder = f'balanced_test_hierarchies'
    files = os.listdir(folder)
    split_func = compo_split if split_type == 'composition' else parent_split
    with open(f'analysis_results/results_{model}.json', 'r') as f:
        model_results = json.load(f)

    for file in files:
        # Setup for splitting
        if file == 'isolated.json':
            continue
        with open(f'{folder}/{file}', 'r') as f:
            data = json.load(f)

        file_splits = {0 : {}, 1 : {}, 2 : {}, 3 : {}, 4 : {}, 5: {}, "any" : {}}
        seen_q = set()

        # Populate the splits
        for key, value in data.items():
            subquestions = value['subquestion']
            hierarchy = value['hierarchy']
            split_func(hierarchy, subquestions, seen_q, model_results, file_splits, only_incorrect)

        # Save the splits
        for key, compositions in file_splits.items():
            for composition, split_data in compositions.items():
                if len(split_data) == 0:
                    continue

                split_folder = f'error_analysis_folders/{model}/{split_type}/{composition}_{key}'
                if only_incorrect:
                    split_folder += "_incorrect"
                mkdir(split_folder)

                with open(f'{split_folder}/{file}', 'w') as f:
                    json.dump(split_data, f)

def compo_split(hierarchy, subquestions, seen_q, model_results, file_splits, only_incorrect):
    for q, comp_subqs in hierarchy.items():
        # Avoid double counting
        if q in seen_q:
            continue
        seen_q.add(q)

        # Skip banned parents
        info = subquestions[q]
        answer = info['answer']
        prediction = get_pred(info, model_results)
        if answer == None or info['type'] in banned_qtypes:
            continue
        allow_parent = (not only_incorrect) or answer != prediction        

        for composition, subqs in comp_subqs.items():
            for key, comp_dict in file_splits.items():
                if composition not in comp_dict:
                    comp_dict[composition] = []

            new_value = {'subquestions' : {}, 'hierarchy' : {}}    
            new_value['subquestions'][q] = info
            new_value['hierarchy'][q] = {composition : subqs}

            total = 0
            correct = 0
            for subq in subqs:                
                subq_info = subquestions[subq]
                new_value['subquestions'][subq] = subq_info

                subq_answer = subq_info['answer']
                if subq_answer == None or subq_info["type"] in banned_qtypes:
                    continue

                total += 1
                if subq_answer == get_pred(subq_info, model_results):
                    correct += 1
                        
            if total != 0 and allow_parent:
                wrong = total - correct
                file_splits[wrong][composition].append(new_value)
                if wrong > 0:
                    file_splits["any"][composition].append(new_value)                            

def parent_split(hierarchy, subquestions, seen_q, model_results, file_splits, only_incorrect):
    for q, comp_subqs in hierarchy.items():
        # Avoid double counting
        if q in seen_q:
            continue
        seen_q.add(q)

        # Skip banned parents
        info = subquestions[q]
        answer = info['answer']
        prediction = get_pred(info, model_results)
        if answer == None or info['type'] in banned_qtypes:
            continue
        allow_parent = (not only_incorrect) or answer != prediction        

        parent_type = collapsed_qtypes[info['type']]
        for key, comp_dict in file_splits.items():
            if parent_type not in comp_dict:
                comp_dict[parent_type] = []

        # Get all child questions
        child_questions = set()
        for composition, subqs in comp_subqs.items():
            for subq in subqs:
                child_questions.add(subq)

        new_value = {"subquestions" : {}, 'hierarchy' : {}}
        new_value['subquestions'][q] = info
        new_value['hierarchy'][q] = comp_subqs

        # Compute as normal
        total = 0
        correct = 0
        for subq in child_questions:
            subq_info = subquestions[subq]
            new_value['subquestions'][subq] = subq_info

            subq_answer = subq_info['answer']
            if subq_answer == None or subq_info["type"] in banned_qtypes:
                continue

            total += 1
            if subq_answer == get_pred(subq_info, model_results):
                correct += 1

        if total != 0 and allow_parent:
            wrong = total - correct
            file_splits[wrong][parent_type].append(new_value)
            if wrong > 0:
                file_splits["any"][parent_type].append(new_value)                                        
        
if __name__ == "__main__":
    split_types = ['composition', 'parent']

    # TODO: MODEL NAME. Replace the names in this list with the names of the models to evaluate
    models = ['hcrn', 'hme', 'psac']
    only_incorrect = False

    for split_type in split_types:
        for model in models:
            print(split_type, model)
            split_dataset(model, split_type, only_incorrect)

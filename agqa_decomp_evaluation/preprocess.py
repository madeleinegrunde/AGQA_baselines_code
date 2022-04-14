import json, os
from utils import mkdir

def get_results(model):
    '''
    Saves a dictionary mapping question q_ids to model predictions, which
    is then used during evaluation.
    '''
    pred_name = f'model_preds/{model}_preds.json'
    with open(pred_name, 'r') as f:
        data = json.load(f)

    results = {}
    for result in data:
        csv_q_id = result['csv_q_id']
        results[csv_q_id] = result['prediction']

    with open(f'analysis_results/results_{model}.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    # TODO: MODEL NAME. Replace the names in this list with the names of the models to evaluate
    models = ['hcrn']

    mkdir('analysis_results')
    for model in models:
        print(f'Processing predictions for {model}')
        get_results(model)

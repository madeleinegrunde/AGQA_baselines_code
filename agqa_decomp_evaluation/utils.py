import os

def mkdir(directory):
    '''
    Produces the supplied directory.

    Arguments:
    - directory (str): The filepath of the directory to produce.
    '''
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

def get_pred(info, model_results):
    '''
    Retrieves the model's prediction from the supplied info dict.

    Arguments:
    - info (dict): A dictionary associated with a single question
                   in the dataset. Must contain the key 'key'
    - model_results (dict): A mapping from question key to predictions.
    '''
    # Get question id
    q_id = info['key']
    return model_results[q_id]

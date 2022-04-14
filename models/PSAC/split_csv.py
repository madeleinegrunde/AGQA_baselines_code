import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='balanced')
    args = parser.parse_args()

    # Determine split index
    df = pd.read_csv('data/dataset/Train_frameqa_question-%s.csv' % args.metric) # TODO: PATH
    split = int(0.9 * df.shape[0])

    # Ensure video data is not split
    while df.loc[split-1, 'vid_id'][:5] == df.loc[split, 'vid_id'][:5]:
        split += 1

    # Split
    df_tr = df.loc[:split-1, :]
    df_tr.to_csv('data/dataset/Train_frameqa_question-%s.csv' % args.metric) # TODO: PATH
    df_val = df.loc[split:, :]
    df_val.to_csv('data/dataset/Valid_frameqa_question-%s.csv' % args.metric) # TODO: PATH
    

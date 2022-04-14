import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='balanced')
    args = parser.parse_args()

    te = pd.read_csv('Test_frameqa_question-%s.csv' % args.metric)
    split = te.shape[0] // 4

    te_1 = te.loc[:split-1]
    te_1.to_csv('Test_frameqa_question-%s_1.csv' % args.metric)

    te_2 = te.loc[split:2*split - 1]
    te_2.to_csv('Test_frameqa_question-%s_2.csv' % args.metric)

    te_3 = te.loc[2*split:3*split - 1]
    te_3.to_csv('Test_frameqa_question-%s_3.csv' % args.metric)

    te_4 = te.loc[3*split:]
    te_4.to_csv('Test_frameqa_question-%s_4.csv' % args.metric)

    assert((te_1.shape[0] + te_2.shape[0] + te_3.shape[0] + te_4.shape[0]) == te.shape[0])
    

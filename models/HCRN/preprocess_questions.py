import argparse
import numpy as np
import os

from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tgif-qa', choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt',
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='data/{}/{}_{}_questions.pt')
    parser.add_argument('--vocab_json', type=str, default='data/{}/{}_vocab.json')
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    parser.add_argument('--question_type', choices=['frameqa', 'action', 'transition', 'count', 'none'], default='none')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--metric', type=str, default='all')


    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset == 'tgif-qa': # TODO: PATH
        if args.mode == 'train':
            args.annotation_file = '~/storage/questions_csv/Train_frameqa_question-%s.csv' % args.metric
        elif args.mode == 'test':
            args.annotation_file = '~/storage/questions_csv/Test_frameqa_question-%s.csv' % args.metric
        else:
            print("invalid mode", args.mode)
        args.output_pt = '/home/ubuntu/storage/questions/tgif-qa_{}_{}_questions-%s.pt' % args.metric
        args.vocab_json = '/home/ubuntu/storage/questions/tgif-qa_{}_vocab-%s.json' % args.metric
        # check if data folder exists
        if not os.path.exists('data/tgif-qa/{}'.format(args.question_type)):
            os.makedirs('data/tgif-qa/{}'.format(args.question_type))

        if args.question_type in ['frameqa', 'count']:
            tgif_qa.process_questions_openended(args)
        else:
            tgif_qa.process_questions_mulchoices(args)
    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/msrvtt/annotations/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msrvtt_qa.process_questions(args)
    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/msvd/MSVD-QA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msvd_qa.process_questions(args)

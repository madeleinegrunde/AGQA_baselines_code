import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset,load_dictionary
from train import train
import utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from models import FrameQA_model
from models import Count_model
from models import Trans_model
from models import Action_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--task', type=str, default='FrameQA',help='FrameQA, Count, Action, Trans')
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--model', type=str, default='temporalAtt', help='temporalAtt')
    parser.add_argument('--max_len',type=int, default=20)
    parser.add_argument('--char_max_len', type=int, default=15)
    parser.add_argument('--num_frame', type=int, default=36)
    parser.add_argument('--output', type=str, default='saved_models/%s/exp-11')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--sentense_file_path',type=str, default='data/dataset')
    parser.add_argument('--glove_file_path', type=str, default='data/glove/glove.6B.300d.txt')
    parser.add_argument('--feat_category',type=str,default='resnet')
    parser.add_argument('--feat_path',type=str,default='../../HME') # TODO: PATH
    parser.add_argument('--Multi_Choice',type=int, default=5)
    parser.add_argument('--vid_enc_layers', type=int, default=1)
    parser.add_argument('--test_phase', type=bool, default=False)
    # ////////////
    parser.add_argument('--metric', type=str, default='balanced')
    # //////////// added ^ metric
    parser.add_argument('--load_saved', type=int, default=0, help='Set equal to 1 to load the most recent model')
    parser.add_argument('--dropout', type=float, default=0.15) # NEW DROPOUT
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight_decay', type=float, default=0.000005)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    print('parameters:', args)
    print('task:',args.task,'model:', args.model)

    # dictionary = Dictionary.load_from_file('./dictionary.pkl')
    dictionary = load_dictionary(args.sentense_file_path, args.task, args.metric)
    # //////////// ^^ added metric to this argument and ones below

    train_dset = VQAFeatureDataset(args, dictionary, args.sentense_file_path,args.feat_category,args.feat_path, mode='Train', metric=args.metric)

    val_dset = VQAFeatureDataset(args, dictionary, args.sentense_file_path,args.feat_category,args.feat_path, mode='Valid', metric=args.metric)
    #eval_dset = VQAFeatureDataset(args, dictionary, args.sentense_file_path,args.feat_category,args.feat_path, mode='Test', metric=args.metric)
    batch_size = args.batch_size

    model_name = args.task+'_model'
    model = getattr(locals()[model_name], 'build_%s' % args.model)(args.task, args.vid_enc_layers, train_dset,
                                                                   args.num_hid, dictionary, args.glove_file_path,
                                                                   dropout=args.dropout).cuda() # NEW DROPOUT

    print('========start train========')
    model = model.cuda()
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=1)
    #eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)

    if args.test_phase:
        # MODEL CHANGE
        #ckpt = 0# load model here
        ckpt = 'saved_models/FrameQA/exp-11/model.pth'
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        eval_score = model.evaluate(eval_loader, eval_dset, test=True)[0]
        print("EVAL SCORE", eval_score)
    else:
        if args.load_saved == 1:
            print('Loading most recent model')
            ckpt = 'saved_models/FrameQA/exp-11/model-current.pth'
            model.load_state_dict(torch.load(ckpt))

        train(model, args, val_dset, dictionary, train_loader, val_loader, args.epochs, args.output)

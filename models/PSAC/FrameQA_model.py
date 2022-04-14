import torch.nn as nn
from models.language_model import *
from models.classifier import SimpleClassifier
from models.fc import FCNet
import torch
from torch.autograd import Variable
from models.attention import NewAttention
from models.attention import *
from torch.nn.utils.weight_norm import weight_norm
import time
import numpy as np
from models.model_utils import *
import torch.nn.functional as F
import json
from dataset import Dictionary, VQAFeatureDataset,load_dictionary

class FrameQAModel(nn.Module):
    def __init__(self, model_name, vid_encoder, ques_encoder,classifier,
                 n_layer=6, n_head=8, d_k=64, d_v=64, v_len=35, v_emb_dim=300,
                 d_model=512, d_inner_hid=512, dropout=0.1, conv_num=4, num_choice=1):
        super(FrameQAModel, self).__init__()

        self.model_name= model_name
        self.num_choice = num_choice
        self.vid_encoder = vid_encoder
        self.ques_encoder = ques_encoder
        self.classifier = classifier
        self.dropout = nn.Dropout(dropout) # NEW DROPOUT

    def forward(self, v, q_w, q_c, labels, return_attns=False):
        # visual info
        vid_output = v + self.vid_encoder(v)  # v : batch_size x v_len x ctx_dim  # batch_size x v_len x d_v
        vid_output = self.dropout(vid_output) # NEW DROPOUT
        fus_output = self.ques_encoder(vid_output, q_w, q_c)
        logits = self.classifier(fus_output)
        out = F.log_softmax(logits, dim=1)
        return out

    def evaluate(self, dataloader, dataset, test=False):
        score = 0
        num_data = 0
        results = []
        j = 0
        for v, q_w, q_c, a, ques_eng, ans_eng, idx, key in iter(dataloader):
            with torch.no_grad():
                v = Variable(v).cuda()
                q_w = Variable(q_w).cuda()
                q_c = Variable(q_c).cuda()
                pred = self.forward(v, q_w, q_c, None)
                # changed to recieve pred_y
                batch_score, preds = compute_score_with_logits(pred, a.cuda())

            score += batch_score
            num_data += pred.size(0)
            prediction = torch.max(pred, 1)[1]

            if j % 100 == 0:
                print('%s/%s' % (j, len(iter(dataloader))))

            # added in lines below, but copied from sample() function
            prediction = np.array(prediction.cpu().data)
            if test:
                for ques, pred, ans, idx, q_id in zip(ques_eng, list(prediction), ans_eng, idx, key):
                    ins = {}
                    ins['index'] = j
                    ins['csv_q_id'] = q_id
                    ins['id'] = int(idx)
                    ins['question'] = ques
                    ins['prediction'] = dataset.label2ans[pred]
                    ins['answer'] = ans
                    ins['prediction_success'] = bool(ans == dataset.label2ans[pred])
                    results.append(ins)
                    j += 1
            else:
                j += len(ques_eng)
        if test:
            print('results 0', results[0])    

            with open('data/prediction/FrameQA_prediction.json', 'w') as f:
                json.dump(results, f)

        score = float(score) / len(dataloader.dataset)
        return score, preds

    def sample(self, dataset, dataloader):
        import json
        score = 0
        num_data = 0
        j = 0
        results = []
        for v, q_w, q_c, a, ques_eng, ans_eng, idx, key in iter(dataloader):
            v = Variable(v).cuda()
            q_w = Variable(q_w).cuda()
            q_c = Variable(q_c).cuda()
            pred = self.forward(v, q_w, q_c, None)
            batch_score, preds = compute_score_with_logits(pred, a.cuda())
            prediction = torch.max(pred, 1)[1]
            score += batch_score
            num_data += pred.size(0)

            # write in json
            prediction = np.array(prediction.cpu().data)
            for ques, pred, ans, idx in zip(ques_eng, list(prediction), ans_eng, idx):
                ins = {}
                ins['index'] = j
                ins['id'] = int(idx)
                ins['question'] = ques
                ins['prediction'] = dataset.label2ans[pred]
                ins['answer'] = ans
                ins['prediction_success'] = bool(ans == dataset.label2ans[pred])
                results.append(ins)
                j += 1
        with open('data/prediction/FrameQA_prediction.json', 'w') as f:
            json.dump(results, f)

        with open('data/prediction/FrameQA_just_pred_y.json', 'w+') as f:
            json.dump(preds, f)


        score = float(score) / len(dataloader.dataset)
        return score

def build_temporalAtt(task_name, n_layer, dataset, num_hid, dictionary, glove_file, dropout=0.1): # NEW DROPOUT
    vid_encoder = Encoder(n_layer=n_layer, n_head=8, d_k=256, d_v=256, v_len=36, v_emb_dim=300,
                               d_model=2048, d_inner_hid=512, dropout=dropout) # NEW DROPOUT: Dropout originally was 0.1
    w = WordEmbedding(dictionary.ntoken, dictionary.c_ntoken, 300, 64, 0.1)
    word_mat, char_mat = w.init_embedding(dictionary, glove_file, task_name)
    ques_encoder = Ques_Encoder(word_mat, char_mat)
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return FrameQAModel(task_name, vid_encoder, ques_encoder, classifier)



def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]
    pred_y = logits.data.cpu().numpy().squeeze()
    target_y = labels.cpu().numpy().squeeze()
    scores = sum(pred_y==target_y)
    # //////////// changed to pass pred_y back up
    return scores, pred_y

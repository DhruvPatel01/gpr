import random
import pickle
import argparse
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import itertools
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_pretrained_bert import BertTokenizer, BertModel
import re
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
import multiprocessing as mp

import pretrained
import dataset
import lee_wo_attn as lee

dev = 'cuda'
BERT_DIM = 1024
HIDDEN_DIM = 256

# def propose(candidates, old_1, old_2):
#     while True:
#         proposal = random.choice(candidates)
#         if any(map(lambda x: x in proposal, old_1)):
#             continue #restrict A: Jon Adams-Wick  B: John Wick
#         if any(map(lambda x: x in proposal, old_2)):
#             continue
#         proposal = proposal.split()
#         if any(map(lambda x: x.isdecimal(), proposal)):
#             continue #there are nouns such as 'Academy 1885'
#         return proposal

# class LeeDS(Dataset):
#     def __init__(self, pkl, tsv_path, layer=-1):
#         dev = 'cpu'
#         tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=True)
#         df = pd.read_csv(tsv_path, '\t')
#         if isinstance(pkl, np.ndarray):
#             data = pkl
#             if layer >= 0:
#                 data = data[layer]
#         else:
#             with open(pkl, 'rb') as f:
#                 data = pickle.load(f)[layer]
#         a_mat = torch.zeros((len(df), BERT_DIM), dtype=torch.float32)
#         b_mat = torch.zeros_like(a_mat)
#         p_mat = torch.zeros((len(df), BERT_DIM), dtype=torch.float32)
#         self.y = [0]*len(df)
#
#         for i, row in df.iterrows():
#             a = torch.from_numpy(np.array(data[i][0]))
#             b = torch.from_numpy(np.array(data[i][1]))
#             c = torch.from_numpy(np.array(data[i][2]))
#             a_mat[i] = a.mean(0)
#             b_mat[i] = b.mean(0)
#             p_mat[i] = c
#
#             if row['A-coref']:
#                 self.y[i] = 0
#             elif row['B-coref']:
#                 self.y[i] = 1
#             else:
#                 self.y[i] = 2
#         self.a_mat = a_mat.to(dev)
#         self.b_mat = b_mat.to(dev)
#         self.p_mat = p_mat.to(dev)
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, i):
#         return (self.a_mat[i], self.b_mat[i], self.p_mat[i],
#                 self.y[i])

class Model(nn.Module):
    def __init__(self, dropout=.4):
        super().__init__()
        # self.bert_weights = torch.randn(24, requires_grad=True).to(dev)
        self.lstm = nn.LSTM(BERT_DIM, HIDDEN_DIM, 1)
        # self.size_embd = nn.Embedding(32, size_dim)
        # self.dist_embd = nn.Embedding(9, dist_dim)
        self.dropout = nn.Dropout(.2)
        self.linear = nn.Sequential(
            # nn.Linear((BERT_DIM*3)+size_dim+(BERT_DIM)*2+dist_dim, 64),
            # nn.Linear(BERT_DIM*3, 64),
            nn.Linear(HIDDEN_DIM*3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            # nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(32),
            nn.Linear(64, 1),
        )

    def forward(self, embd, spans):
        # dev = 'cuda'
        # temp = torch.zeros_like(embd[0]).to(dev)
        # for i in range(embd.shape[0]):
        #     temp += self.bert_weights[i] * embd[i]
        # embd = temp/embd.shape[0]

        self.lstm.flatten_parameters()
        outs, (hn, cn) = self.lstm(embd.permute(1, 0, 2))
        embd = outs.permute(1, 0, 2)
        a_mask = torch.arange(embd.size(1)).to(dev)
        b_mask = torch.arange(embd.size(1)).to(dev)
        a_mask = a_mask.expand(embd.size(0), -1)
        b_mask = b_mask.expand(embd.size(0), -1)
        a_mask = (a_mask < spans[:, 1].view(-1, 1)) & ~(a_mask < spans[:, 0].view(-1, 1))
        b_mask = (b_mask < spans[:, 3].view(-1, 1)) & ~(b_mask < spans[:, 2].view(-1, 1))
        a_mask, b_mask = a_mask.to(torch.float32).unsqueeze(-1), b_mask.to(torch.float32).unsqueeze(-1)

        a, b = (embd * a_mask).sum(1), (embd * b_mask).sum(1)
        a /= (spans[:, 1] - spans[:, 0]).to(torch.float32).view(-1, 1)
        b /= (spans[:, 3] - spans[:, 2]).to(torch.float32).view(-1, 1)
        p = embd[range(embd.size(0)), spans[:, 4]]

        a = self.dropout(a)
        b = self.dropout(b)
        p = self.dropout(p)

        ap = a*p#[:, -BERT_DIM:] * p
        bp = b*p#[:, -BERT_DIM:] * p
        a = torch.cat([a, p, ap], -1)
        b = torch.cat([b, p, bp], -1)

        ha = self.linear(a)
        hb = self.linear(b)
        return torch.cat([ha, hb, torch.zeros_like(ha)], -1)

def train(tdl, vdl, args):
    # dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert = nn.DataParallel(BertModel.from_pretrained('bert-large-cased')).to(dev)
    model = nn.DataParallel(Model(dropout=.5)).to(dev)
    # bert = BertModel.from_pretrained('bert-large-cased').to(dev)
    # model = Model(dropout=.5).to(dev)
    if args.pre_model:
        model.load_state_dict(torch.load(args.logdir + '/' + args.pre_model))
    loss_fn = nn.CrossEntropyLoss()#torch.tensor([0.15, 0.15, .7]).to(dev))
    # optim = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=.001)
    optim = torch.optim.Adam(model.parameters(), weight_decay=0.001)
    if args.pre_opt:
        optim.load_state_dict(torch.load(args.logdir + '/' + args.pre_opt))
    writer = SummaryWriter(args.logdir+'/Log')

    best_val_loss = 100
    last_updated = 0
    total_loss = 0
    for epoch in tqdm.trange(100):
        model.train()
        for elem in tqdm.tqdm(tdl, "Epoch %d" % epoch):
            sent, spans, y = [e.to(dev) for e in elem]
            with torch.no_grad():
                embd, _ = bert(sent)
            # for i in range(len(embd)):
            #     embd[i] = embd[i].cpu().detach().numpy()
            # embd = torch.from_numpy(np.array(embd)).to(dev)
            embd = embd[args.bert_layer]

            optim.zero_grad()
            yp = model(embd, spans)
            loss = loss_fn(yp, y)
            total_loss += float(loss)
            loss.backward()
            optim.step()
        total_loss /= len(tdl)
        writer.add_scalar('train/loss', total_loss, epoch)

        model.eval()
        accu = 0.
        totl = 0
        with torch.no_grad():
            for elem in tqdm.tqdm(vdl, "Epoch %d" % epoch):
                sent, spans, y = [e.to(dev) for e in elem]
                # with torch.no_grad():
                embd, _ = bert(sent)
                # for i in range(len(embd)):
                #     embd[i] = embd[i].cpu().detach().numpy()
                # embd = torch.from_numpy(np.array(embd)).to(dev)

                embd = embd[args.bert_layer]
                y = y.to(dev)
                yp = model(embd, spans)
                l = F.cross_entropy(yp, y, reduction='none')
                loss += float(l.sum())
                yp = torch.softmax(yp, -1).argmax(-1)
                accu += float((yp == y).sum())
                totl += len(y)
            # for a,b,p,y,da,db,sa,sb in vdl:
            #     a, b, p  = a.to(dev), b.to(dev), p.to(dev)
            #     yp = model(a, b, p)
            accu = (1. * accu)/totl
            loss /= totl
            writer.add_scalar('valid/loss', float(loss), epoch)
            writer.add_scalar('valid/accuracy', float(accu), epoch)
            best = ' '
            if best_val_loss > loss:
                last_updated = 0
                best_val_loss = loss
                torch.save(model.state_dict(), args.logdir+'/best_model-%.2f.pt'%best_val_loss)
                torch.save(optim.state_dict(), args.logdir+'/best_optim-%.2f.pt'%best_val_loss)
                best = '*'
            else:
                last_updated += 1
                if last_updated > args.patience:
                    return model
            tqdm.tqdm.write("Epoch: %d, Tr.loss: %.4f  Vl.loss: %.4f %s" % (epoch, total_loss, loss, best))

def test(ds, args, labels=True, out='test_log.txt'):
    # dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    bert = nn.DataParallel(BertModel.from_pretrained('bert-large-cased')).to(dev)
    model = nn.DataParallel(Model(dropout=.5)).to(dev)
    # bert = BertModel.from_pretrained('bert-large-cased').to(dev)
    # model = Model(dropout=.5).to(dev)
    model.load_state_dict(torch.load(args.logdir + '/' + args.pre_model))

    log_loss = 0
    log_file = open(args.logdir + '/' + out, 'w')
    print("ID,A,B,NEITHER", file=log_file)
    model.eval()
    with torch.no_grad():
        for elem in tqdm.tqdm(ds, "Testing"):
            if labels:
                sent, spans, y, name = elem
            else:
                sent, spans, name = elem
            sent = sent.to(dev).unsqueeze(0)
            spans = spans.to(dev).unsqueeze(0)
            embd, _ = bert(sent)
            embd = embd[args.bert_layer]
            # a_mask = torch.arange(embd.size(1)).to(dev)
            # b_mask = torch.arange(embd.size(1)).to(dev)
            # a_mask = a_mask.expand(embd.size(0), -1)
            # b_mask = b_mask.expand(embd.size(0), -1)
            # a_mask = (a_mask < spans[:, 1].view(-1, 1)) & ~(a_mask < spans[:, 0].view(-1, 1))
            # b_mask = (b_mask < spans[:, 3].view(-1, 1)) & ~(b_mask < spans[:, 2].view(-1, 1))
            # a_mask, b_mask = a_mask.to(torch.float32).unsqueeze(-1), b_mask.to(torch.float32).unsqueeze(-1)
            #
            # a, b = (embd * a_mask).sum(1), (embd * b_mask).sum(1)
            # a /= (spans[:, 1] - spans[:, 0]).to(torch.float32).view(-1, 1)
            # b /= (spans[:, 3] - spans[:, 2]).to(torch.float32).view(-1, 1)
            # p = embd[range(embd.size(0)), spans[:, 4]]
            logits = model(embd, spans)
            prob = tuple(torch.softmax(logits, -1).cpu()[0])
            print('%s,%.4f,%.4f,%.4f' % (name, *prob), file=log_file)
            if labels:
                logprobs = torch.log_softmax(logits, -1)
                log_loss -= float(logprobs[0, y])
    print("Loss: %.4f"%(log_loss/len(ds)))
    log_file.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_layer', '-l', '--bl', default=19, type=int)
    parser.add_argument('--patience', '-p', default=10, type=int)
    parser.add_argument('--batch_size', '--bs', default=10, type=int)
    parser.add_argument('-w', action='store_true', help='include winobias')
    parser.add_argument('-d', action='store_true', help='include defpr')
    parser.add_argument('--logdir', default='.')
    parser.add_argument('--pre_model')
    parser.add_argument('--pre_opt')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test2', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=False)

    if args.test:
        ds = dataset.DS_Augmented('./data/test_stage_1.tsv',
                                  tokenizer, 0, return_ids=True)
        test(ds, args, True, out='test_stage_1.csv')
    elif args.test2:
        ds = dataset.DS_Augmented('./data/test_stage_2.tsv',
                                  tokenizer, 0, False, return_ids=True)
        test(ds, args, False, out='test_stage_2.csv')
    else:
        # vds = lee.DS('./data/embed-bert-19.pkl', './data/gap-validation.tsv', 0)
        vds = dataset.DS_Simple('./data/gap-coreference/gap-validation-corrected.tsv', tokenizer)
        vdl = DataLoader(vds, 5, True, collate_fn=vds.collate_fn)

        ds = dataset.DS_Augmented('./data/gap-coreference/gap-development-corrected.tsv', tokenizer)
        tds = ds
        if args.d:
            defds = dataset.DS_Augmented('./data/defpr/defpr.csv', tokenizer, 0)
            tds = ConcatDataset([tds, defds])
        tdl = DataLoader(tds, args.batch_size, True, collate_fn=ds.collate_fn)

        train(tdl, vdl, args)

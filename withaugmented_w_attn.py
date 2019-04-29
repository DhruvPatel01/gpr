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
import withaugmented

BERT_DIM = 1024

class MultiAttention(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.lin1 = nn.Linear(2*BERT_DIM, h_dim)
        self.lin2 = nn.Linear(h_dim, 1)

    def forward(self, sent, p, a_mask, b_mask):
        # BxBERT_DIm -> BxLxBERT_DIM
        p = p.unsqueeze(1).expand(-1, sent.size(1), -1)

        # dim = BxLx2*BERT_DIM
        x = torch.cat([sent, p], dim=-1)
        #dim = BxLxh_dim
        h = torch.tanh(self.lin1(x))

        #dim = BxL
        h = self.lin2(h).squeeze(-1)

        a = torch.softmax(h.masked_fill(~a_mask, float('-inf')), -1)
        b = torch.softmax(h.masked_fill(~b_mask, float('-inf')), -1)
        a, b = a.unsqueeze(1), b.unsqueeze(1)
        a = torch.bmm(a, sent).squeeze(1)
        b = torch.bmm(b, sent).squeeze(1)
        return a, b

def generate_masks(embd, spans, dev='cpu'):
    a_mask = torch.arange(embd.size(1)).to(dev)
    b_mask = torch.arange(embd.size(1)).to(dev)
    a_mask = a_mask.expand(embd.size(0), -1)
    b_mask = b_mask.expand(embd.size(0), -1)
    a_mask = (a_mask < spans[:, 1].view(-1, 1)) & ~(a_mask < spans[:, 0].view(-1, 1))
    b_mask = (b_mask < spans[:, 3].view(-1, 1)) & ~(b_mask < spans[:, 2].view(-1, 1))
    return a_mask, b_mask

def train(tdl, vdl, args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    bert = BertModel.from_pretrained('bert-large-cased').to(dev)
    attn = MultiAttention(256).to(dev)
    if args.pre_attn:
        attn.load_state_dict(torch.load(args.logdir + '/' + args.pre_attn))
    model = withaugmented.Model(dropout=.5).to(dev)
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
            embd = embd[args.bert_layer]
            a_mask, b_mask = generate_masks(embd, spans, dev)
            p = embd[range(embd.size(0)), spans[:, 4]]
            a, b = attn(embd, p, a_mask, b_mask)

            optim.zero_grad()
            yp = model(a, b, p)
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
            for elem in vdl:
                sent, spans, y = [e.to(dev) for e in elem]
                with torch.no_grad():
                    embd, _ = bert(sent)
                embd = embd[args.bert_layer]
                a_mask, b_mask = generate_masks(embd, spans, dev)
                p = embd[range(embd.size(0)), spans[:, 4]]
                a, b = attn(embd, p, a_mask, b_mask)

                yp = model(a, b, p)
                l = F.cross_entropy(yp, y, reduction='none')
                yp = torch.softmax(yp, -1).argmax(-1)
                loss += float(l.sum())
                accu += float((yp == y).sum())
                totl += len(y)
            accu = (1. * accu)/totl
            loss /= totl
            writer.add_scalar('valid/loss', float(loss), epoch)
            writer.add_scalar('valid/accuracy', float(accu), epoch)
            best = ' '
            if best_val_loss > loss:
                last_updated = 0
                best_val_loss = loss
                torch.save(model.state_dict(), args.logdir+'/best_model-%.4f.pt'%best_val_loss)
                torch.save(optim.state_dict(), args.logdir+'/best_optim-%.4f.pt'%best_val_loss)
                torch.save(attn.state_dict(), args.logdir+'/best_attn-%.4f.pt'%best_val_loss)
                best = '*'
            else:
                last_updated += 1
                if last_updated > args.patience:
                    return model
            tqdm.tqdm.write("Epoch: %d, Tr.loss: %.4f  Vl.loss: %.4f %s" % (epoch, total_loss, loss, best))

def test(ds, args, labels=True, out='test_log.txt'):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert = BertModel.from_pretrained('bert-large-cased').to(dev)
    attn = MultiAttention(256).to(dev)
    model = withaugmented.Model(dropout=.5).to(dev)
    attn.load_state_dict(torch.load(args.logdir + '/' + args.pre_attn))
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
            a_mask, b_mask = generate_masks(embd, spans, dev)
            p = embd[range(embd.size(0)), spans[:, 4]]
            a, b = attn(embd, p, a_mask, b_mask)
            logits = model(a, b, p)
            prob = tuple(torch.softmax(logits, -1).cpu()[0])
            print('%s,%.4f,%.4f,%.4f' % (name, *prob), file=log_file)
            if labels:
                logprobs = torch.log_softmax(logits, -1)
                log_loss -= float(logprobs[0, y])
    print("Loss: %.4f"%(log_loss/len(ds)))
    log_file.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_layer', '-l', '--bl', default=17, type=int)
    parser.add_argument('--patience', '-p', default=10, type=int)
    parser.add_argument('--batch_size', '--bs', default=6, type=int)
    parser.add_argument('-w', action='store_true', help='include winobias')
    parser.add_argument('-d', action='store_true', help='include defpr')
    parser.add_argument('--logdir', default='.')
    parser.add_argument('--pre_model')
    parser.add_argument('--pre_attn')
    parser.add_argument('--pre_opt')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test2', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=False)

    if args.test:
        ds = dataset.DS_Augmented('./data/gap-coreference/gap-development-corrected.tsv',
                                   tokenizer, 0, return_ids=True)
        test(ds, args, True, out='test_stage_1.csv')
    elif args.test2:
        ds = dataset.DS_Augmented('./data/gap-coreference/test_stage_2.tsv', tokenizer, 0,
                                  False, return_ids=True)
        test(ds, args, False, out='test_stage_2.csv')
    else:
        vds = dataset.DS_Augmented('./data/gap-coreference/gap-validation-corrected.tsv',
                                   tokenizer, 0)
        ds = dataset.DS_Augmented('./data/gap-coreference/gap-test-corrected.tsv', tokenizer)
        tds = ds

        if args.d:
            defds = dataset.DS_Augmented('./data/defpr/defpr.csv', tokenizer, 0)
            tds = ConcatDataset([tds, defds])
        tdl = DataLoader(tds, args.batch_size, True, collate_fn=ds.collate_fn)
        vdl = DataLoader(vds, 5, collate_fn=vds.collate_fn)

        train(tdl, vdl, args)
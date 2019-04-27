import pickle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
from tensorboardX import SummaryWriter
from pytorch_pretrained_bert import BertTokenizer
import pretrained

def bucket_id(x):
    buckets = [1, 2, 3, 4, 8, 16, 32, 64]
    assert x > 0
    return sum(1 for i in buckets if x >= i)-1

BERT_DIM = 1024

class DS(Dataset):
    def __init__(self, pkl, tsv_path, layer=-1, do_lower_case=False):
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=do_lower_case)
        df = pd.read_csv(tsv_path, '\t')
        if isinstance(pkl, np.ndarray):
            data = pkl
            if layer >= 0:
                data = data[layer]
        else:
            with open(pkl, 'rb') as f:
                data = pickle.load(f)[layer]
        self.a_mat = np.zeros((len(df), BERT_DIM), dtype=np.float32)
        self.b_mat = np.zeros_like(self.a_mat)
        self.p_mat = np.zeros((len(df), BERT_DIM), dtype=np.float32)
        self.a_dist = [0]*len(df)
        self.b_dist = [0]*len(df)
        self.a_size = [0]*len(df)
        self.b_size = [0]*len(df)
        self.y = [0]*len(df)

        for i, row in df.iterrows():
            a = np.array(data[i][0])
            b = np.array(data[i][1])
            c = np.array(data[i][2])
            self.a_mat[i] = a.mean(0)
            self.b_mat[i] = b.mean(0)
            # self.a_mat[i] = np.concatenate([a[0], a[-1], a.mean(0)])
            # self.b_mat[i] = np.concatenate([b[0], b[-1], b.mean(0)])
            self.p_mat[i] = c
            self.a_size[i] = len(a)
            self.b_size[i] = len(b)

            if row['A-coref']:
                self.y[i] = 0
            elif row['B-coref']:
                self.y[i] = 1
            else:
                self.y[i] = 2

            p = row['A-offset']
            q = row['Pronoun-offset']
            size = self.a_size[i]
            if p > q:
                p, q = q, p
                size = 1
            subsentence = tokenizer.tokenize(row.Text[p:q])
            self.a_dist[i] = bucket_id(len(subsentence) - size + 1)

            p = row['B-offset']
            q = row['Pronoun-offset']
            size = self.b_size[i]
            if p > q:
                p, q = q, p
                size = 1
            subsentence = tokenizer.tokenize(row.Text[p:q])
            self.b_dist[i] = bucket_id(len(subsentence) - size + 1)

    def __len__(self):
        return len(self.a_dist)

    def __getitem__(self, i):
        return (self.a_mat[i], self.b_mat[i], self.p_mat[i],
                self.y[i],
                self.a_dist[i], self.b_dist[i],
                self.a_size[i], self.b_size[i])

class Model(nn.Module):
    def __init__(self, size_dim=20, dist_dim=20, dropout=.4):
        super().__init__()
        self.size_embd = nn.Embedding(32, size_dim)
        self.dist_embd = nn.Embedding(9, dist_dim)
        self.dropout = nn.Dropout(.2)
        self.linear = nn.Sequential(
            nn.Linear((BERT_DIM)+size_dim+(BERT_DIM)*2+dist_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(32),
            nn.Linear(32, 1),
        )

    def forward(self, a, b, p, dist_a, dist_b, size_a, size_b):
        size_a = self.size_embd(size_a)
        size_b = self.size_embd(size_b)
        dist_a = self.dist_embd(dist_a)
        dist_b = self.dist_embd(dist_b)

        a = self.dropout(a)
        b = self.dropout(b)
        p = self.dropout(p)

        ap = a[:, -BERT_DIM:] * p
        bp = b[:, -BERT_DIM:] * p
        a = torch.cat([a, p, ap, size_a, dist_a], -1)
        b = torch.cat([b, p, bp, size_b, dist_b], -1)

        ha = self.linear(a)
        hb = self.linear(b)
        return torch.cat([ha, hb, torch.zeros_like(ha)], -1)

def parse_args():
    pass

def train(tdl, vdl, patience=5):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(dropout=.5).to(dev)
    loss_fn = nn.CrossEntropyLoss()#torch.tensor([0.15, 0.15, .7]).to(dev))
    # optim = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=.001)
    optim = torch.optim.Adam(model.parameters(), weight_decay=0.001)
    writer = SummaryWriter('./Log')

    best_val_loss = 100
    last_updated = 0
    for epoch in range(100):
        model.train()
        for a,b,p,y,da,db,sa,sb in tdl:
            optim.zero_grad()
            a, b, p = a.to(dev), b.to(dev), p.to(dev)
            sa, sb = sa.to(dev), sb.to(dev)
            da, db = da.to(dev), db.to(dev)
            y = y.to(dev)
            yp = model(a, b, p, da, db, sa, sb)
            loss = loss_fn(yp, y)
            loss.backward()
            optim.step()

        model.eval()
        loss = 0
        accu = 0.
        totl = 0
        with torch.no_grad():
            for a,b,p,y,da,db,sa,sb in tdl:
                a, b, p  = a.to(dev), b.to(dev), p.to(dev)
                sa, sb = sa.to(dev), sb.to(dev)
                da, db = da.to(dev), db.to(dev)
                y = y.to(dev)
                yp = model(a, b, p, da, db, sa, sb)
                l = loss_fn(yp, y)
                yp = torch.softmax(yp, -1).argmax(-1)
                loss += float(l)
                accu = accu + float((yp == y).sum())
                totl += len(y)
            accu = (1. * accu)/totl
            loss /= len(tdl)
            writer.add_scalar('train/loss', float(loss), epoch)
            writer.add_scalar('train/accuracy', float(accu), epoch)
        loss = 0.
        accu = 0.
        totl = 0
        with torch.no_grad():
            for a,b,p,y,da,db,sa,sb in vdl:
                a, b, p  = a.to(dev), b.to(dev), p.to(dev)
                sa, sb = sa.to(dev), sb.to(dev)
                da, db = da.to(dev), db.to(dev)
                y = y.to(dev)
                yp = model(a, b, p, da, db, sa, sb)
                l = F.cross_entropy(yp, y, reduction='none')
                yp = torch.softmax(yp, -1).argmax(-1)
                loss += float(l.sum())
                accu += float((yp == y).sum())
                totl += len(y)
            accu = (1. * accu)/totl
            loss /= totl
            writer.add_scalar('valid/loss', float(loss), epoch)
            writer.add_scalar('valid/accuracy', float(accu), epoch)
            if best_val_loss > loss:
                last_updated = 0
                best_val_loss = loss
                print("Epoch: %d, loss: %.2f" % (epoch, loss))
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                last_updated += 1
                if last_updated > patience:
                    return model


if __name__ == "__main__":
    # data = pretrained.tokenize_tsv('./data/gap-coreference/gap-validation.tsv', list(range(24)), model='bert-large-cased')
    # with open('./data/gap-coreference/pickles/large_val.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    
    for layer in range(24):
        print("========= Layer: ", layer)
        print("Preparing data ...")
        data = pretrained.tokenize_tsv('./data/gap-coreference/gap-test.tsv', [layer], model='bert-large-cased')
        with open('./data/gap-coreference/pickles/large_test_%d.pkl' % layer, 'wb') as f:
            pickle.dump(data, f)

        gprds = DS('./data/gap-coreference/pickles/large_test_%d.pkl' % layer,
                   './data/gap-coreference/gap-test.tsv', layer=0)
        vds = DS('./data/gap-coreference/pickles/large_val.pkl',
                 './data/gap-coreference/gap-validation.tsv', layer=layer)
    # winods = DS('./data/winobias/pickle.pkl', 
    #           './data/winobias/csv/out.csv', layer=0)
    # defprds = DS('./data/defpr/defpr.pkl', 
    #              './data/defpr/defpr.csv', layer=0)
    # tds = ConcatDataset([gprds, winods, defprds])
    
        tdl = DataLoader(gprds, 128, True, num_workers=3)
        vdl = DataLoader(vds, 128)
        print("Tranining ...")
        train(tdl, vdl, patience=10)
        print()
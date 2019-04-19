import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tensorboardX import SummaryWriter

def bucket_id(x):
    assert x > 0
    if 0 < x < 5:
        return x-1
    elif x < 8:
        return 4
    elif x < 16:
        return 5
    elif x < 32:
        return 6
    elif x < 64:
        return 7
    else:
        return 8

class DS(Dataset):
    def __init__(self, a_mat, b_mat, p_mat, tsv_path):
        self.a_mat = a_mat.astype(np.float32)
        self.b_mat = b_mat.astype(np.float32)
        self.p_mat = p_mat.astype(np.float32)
        self.a_dist = [0]*len(a_mat)
        self.b_dist = [0]*len(a_mat)
        self.a_size = [0]*len(a_mat)
        self.b_size = [0]*len(b_mat)
        self.y = [2]*len(b_mat)

        df = pd.read_csv(tsv_path, '\t')
        for i, row in df.iterrows():
            self.a_size[i] = len(row['A'].split())
            self.b_size[i] = len(row['B'].split())

            if row['A-coref']:
                self.y[i] = 0
            elif row['B-coref']:
                self.y[i] = 1

            p = row['A-offset']
            q = row['Pronoun-offset']
            size = self.a_size[i]
            if p > q:
                p, q = q, p
                size = 1
            subsentence = row.Text[p:q]
            self.a_dist[i] = bucket_id(len(subsentence.split()) - size + 1)

            p = row['B-offset']
            q = row['Pronoun-offset']
            size = self.b_size[i]
            if p > q:
                p, q = q, p
                size = 1
            subsentence = row.Text[p:q]
            self.b_dist[i] = bucket_id(len(subsentence.split()) - size + 1)

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
        
        self.linear = nn.Sequential(
            nn.Linear(768+size_dim+768*2+dist_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)#, bias=False),
        )

    def forward(self, a, b, p, dist_a, dist_b, size_a, size_b):
        size_a = self.size_embd(size_a)
        size_b = self.size_embd(size_b)
        dist_a = self.dist_embd(dist_a)
        dist_b = self.dist_embd(dist_b)

        ap = a * p
        bp = b * p
        a = torch.cat([a, size_a, p, ap, dist_a], -1)
        b = torch.cat([b, size_b, p, bp, dist_b], -1)

        ha = self.linear(a)
        hb = self.linear(b)
        return torch.cat([ha, hb, torch.zeros_like(ha)], -1)

def parse_args():
    pass

def train():
    tmat = np.load('./data/gap-coreference/pickles/bert-layer-9-test.npy')
    vmat = np.load('./data/gap-coreference/pickles/bert-layer-9-val.npy')

    amat = tmat[:, :768]
    bmat = tmat[:, 768:1536]
    pmat = tmat[:, 1536:]
    tds = DS(amat, bmat, pmat, './data/gap-coreference/gap-test.tsv')

    amat = vmat[:, :768]
    bmat = vmat[:, 768:1536]
    pmat = vmat[:, 1536:]
    vds = DS(amat, bmat, pmat, './data/gap-coreference/gap-validation.tsv')

    tdl = DataLoader(tds, 128, True, num_workers=3)
    vdl = DataLoader(vds, 128)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(dropout=.6).to(dev)
    loss_fn = nn.CrossEntropyLoss()#torch.tensor([0.15, 0.15, .7]).to(dev))
    optim = torch.optim.Adam(model.parameters(), weight_decay=.005)
    writer = SummaryWriter('./Log')
    
    best_val_loss = 100
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
                best_val_loss = loss
                print("Epoch: %d, loss: %.2f" % (epoch, loss))
                torch.save(model.state_dict(), 'best_model.pt')

if __name__ == "__main__":
    train()
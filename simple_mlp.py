import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tensorboardX import SummaryWriter

class DS(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, inp_dim, h1, h2, dropout=.5):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(inp_dim, h1),
            nn.ELU(),
            nn.BatchNorm1d(h1),
            nn.Dropout(dropout),
        )        

        self.fc2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ELU(),
            nn.BatchNorm1d(h2),
            nn.Dropout(dropout),
        )

        self.fc3 = nn.Linear(h2, 3)

    def forward(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        return self.fc3(h2)

def main():
    tdf = pd.read_csv('./data/gap-coreference/gap-test.tsv', '\t')
    vdf = pd.read_csv('./data/gap-coreference/gap-validation.tsv', '\t')
    ty = 2+np.zeros(len(tdf), dtype=np.uint8)
    vy = 2+np.zeros(len(vdf), dtype=np.uint8)
    ty[tdf['A-coref']] = 0
    ty[tdf['B-coref']] = 1
    vy[vdf['A-coref']] = 0
    vy[vdf['B-coref']] = 1

    tX = np.load('./data/gap-coreference/pickles/bert-layer-9-test.npy')
    vX = np.load('./data/gap-coreference/pickles/bert-layer-9-val.npy')
    # tX1 = np.load('./data/gap-coreference/pickles/bert-layer-9-test-BAP.npz')
    # ty1 = tX1['y']
    # tX1 = tX1['x']
    # tX = np.concatenate([tX, tX1])
    # ty = np.concatenate([ty, ty1])
    tds = DS(tX, ty)
    vds = DS(vX, vy)

    tdl = DataLoader(tds, batch_size=128, shuffle=True)
    vdl = DataLoader(vds, batch_size=256)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP(tX.shape[1], 128, 32, .6).to(dev)
    # w = torch.tensor([0.1, 0.1, 0.8]).to(dev)
    loss = torch.nn.CrossEntropyLoss()#weight=w)
    optim = torch.optim.Adam(model.parameters(), weight_decay=0.005)
    # optim = torch.optim.SGD(model.parameters())
    writer = SummaryWriter('./MLP_Log')
    best_valid = 100
    best_accu = 0
    for e in range(500):
        model.train()
        for x, y in tdl:
            x = x.to(dev)
            y = y.to(dev)
            optim.zero_grad()
            logits = model(x)
            l = loss(logits, y)
            l.backward()
            optim.step()

        model.eval()
        labels = 'train valid'.split()
        for i, dl in enumerate([tdl, vdl]):
            l = 0
            totl = 0
            accu = 0
            for x, y in dl:
                x = x.to(dev)
                y = y.to(dev)
                logits = model(x)
                l += float(loss(logits, y))
                yp = torch.softmax(logits, -1).argmax(-1)
                totl += len(yp)
                accu += int((yp == y).sum())
            accu /= totl
            l /= len(dl)
            writer.add_scalar(f'{labels[i]}/loss', l)
            writer.add_scalar(f'{labels[i]}/accuracy', l)
            print(f'{l:.2f}\t{accu:.2f}', end='\t')
        print(f'\t{best_accu}')
        if l < best_valid:
            best_valid = l
            best_accu = accu
            torch.save(model.state_dict(), './mlp_best.pt')

if __name__ == "__main__":
    main()
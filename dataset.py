import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pathlib
import pickle
import to_window

class DS_v1(Dataset):
    def __init__(self, pickle_dir, embed_dir,
                 max_window=10):
        pickle_path = pathlib.Path(pickle_dir)
        embd_dir = pathlib.Path(embed_dir)
        pickles = pickle_path.glob('*.pkl')
        Xs = []
        Ys = []
        Zs = []
        lengths = []
        tokens = []
        titles = []
        for pkl in pickles:
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            embed = np.load(embd_dir / (pkl.stem + '.npy'))
            pronoun_ind = data['pronoun_index']
            pronoun_vect = embed[pronoun_ind]
            ans_ind = data['ans_index']
            ans = data['ans']
            ys = np.zeros(len(embed), dtype=np.int64)
            ys[ans_ind:ans_ind+len(ans)] = 1
            assert ys[pronoun_ind] == 0
            newx, newy, lens = to_window.to_window_cbow(embed, 
                                                        ans_ind,
                                                        len(ans),
                                                        pronoun_ind,
                                                        len(ans)+3)
            Xs.append(newx)
            Ys.append(newy)
            Zs.append(pronoun_vect)
            tokens.append(data['tokens'])
            titles.append(pkl.stem)
            lengths.append(lens)
        self.Xs = Xs
        self.Ys = Ys
        self.Zs = Zs
        self.lengths = lengths
        self.tokens = tokens
        self.titles = titles

    def __len__(self):
        return len(self.Zs)

    def __getitem__(self, i):
        return self.Xs[i], self.Ys[i], self.Zs[i], self.tokens[i], self.titles[i]

if __name__ == '__main__':
    ds = DS_v1('data/gap/pickles/dev', 'data/gap/embeds')


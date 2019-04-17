import torch
import tqdm
import numpy as np
import pandas as pd
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

class DS_v2(Dataset):
    """Dataset intended for windows of threes."""
    def __init__(self, pickle_dir, embed_dir):
        pickle_path = pathlib.Path(pickle_dir)
        embd_dir = pathlib.Path(embed_dir)
        pickles = pickle_path.glob('*.pkl')
        Xs = []
        Ys = []
        Zs = []
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

            ys = np.zeros(len(embed), dtype=np.float32)
            ys[ans_ind:ans_ind+len(ans)] = 1
            # assert ys[pronoun_ind] == 0
            Xs.append(embed)
            Ys.append(ys)
            Zs.append(pronoun_vect)
            tokens.append(data['tokens'])
            titles.append(pkl.stem)
        self.Xs = Xs
        self.Ys = Ys
        self.Zs = Zs
        self.tokens = tokens
        self.titles = titles

    def __len__(self):
        return len(self.Zs)

    def __getitem__(self, i):
        return self.Xs[i], self.Ys[i], self.Zs[i], self.tokens[i], self.titles[i]

class Dataset_flair_mean(Dataset):
    """Dataset to be used with flair library"""
    def __init__(self, tsv_file, pkl_file, embedder, k_neg=2, is_train=True,
                 ignore_blank=True):
        """
        Args:
            tsv_file: gap_file
            pkl_file: file containing list of flair Sentences
            embedder: object with embed method that can be applied on Sentences
            is_train: if true treat as training dataset, else runtime ds.
                datasets with answers are considered training dataset.
        """
        super().__init__()
        with open(pkl_file, 'rb') as f:
            sents = pickle.load(f)

        df = pd.read_csv(tsv_file, '\t', index_col='ID')
        self.k_neg = k_neg
        pronouns = []
        nouns = []
        answs = []
        As = []
        Bs = []
        ternary_ans = []
        for i in tqdm.tqdm(range(len(sents)), total=len(sents)):
            sent = sents[i]
            embedder.embed(sent)
            sent_spans = sent.get_spans('ner')
            sent_spans = list(filter(lambda x: x.tag == 'PER', sent_spans))
            if len(sent_spans) == 0:
                continue
            
            row = df.iloc[i]
            ans_offset = 10000
            a_offset = row['A-offset']
            b_offset = row['B-offset']

            if row['A-coref']:
                ans_offset = row['A-offset']
            elif row['B-coref']:
                ans_offset = row['B-offset']

            if ignore_blank and ans_offset == 10000:
                continue

            if row['A-coref']:
                ternary_ans.append(0)
            elif row['B-coref']:
                ternary_ans.append(1)
            else:
                ternary_ans.append(2)                

            tokens = sent.tokens
            pronound_indx = row['Pronoun-offset']
            
            for token in tokens:
                if token.start_pos <= pronound_indx < token.end_pos:
                    pronouns.append(token)
                    break
            else:
                raise Exception("no pronoun found")

            
            nouns.append([])
            noun_inserted = False
            a_inserted = False
            b_inserted = False
            for span in sent_spans:
                assert span.tag == 'PER'
                if span.start_pos <= ans_offset < span.end_pos:
                    assert noun_inserted == False
                    answs.append(span.tokens)
                    noun_inserted = True
                    if not is_train:
                        nouns[-1].append(span.tokens)    
                else:
                    nouns[-1].append(span.tokens)

                if span.start_pos <= a_offset < span.end_pos:
                    As.append(span)
                    a_inserted = True
                if span.start_pos <= b_offset < span.end_pos:
                    As.append(span)
                    b_inserted = True
                    
            if not noun_inserted:
                answs.append(None)
            if not a_inserted:
                As.append(None)
            if not b_inserted:
                Bs.append(None)

        self.pronouns = pronouns
        self.spans = nouns
        self.answs = answs
        self.sents = sents
        self.training = is_train
        self.ternary = ternary_ans
        self.As = As
        self.Bs = Bs

    def __len__(self):
        return len(self.pronouns)

    def __getitem__(self, i):
        prn = self.pronouns[i]
        prn = prn.embedding
        spn = self.spans[i]
        spn = [sum(tok.embedding for tok in span)/len(span) for span in spn]
        if self.training:
            ans = sum(tok.embedding for tok in self.answs[i])/len(self.answs[i])
            if len(spn) > self.k_neg:
                indxs = np.random.choice(len(spn), self.k_neg)
                spn = [spn[i] for i in indxs]
            spn.append(ans)
            spn = torch.stack(spn)
            lbl = torch.zeros(len(spn), dtype=torch.int64)
            lbl[-1] = 1
            return prn, spn, lbl 
        spn = torch.stack(spn)        
        return prn, spn

if __name__ == '__main__':
    ds = DS_v1('data/gap/pickles/dev', 'data/gap/embeds')


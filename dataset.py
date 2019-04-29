import random, itertools, re
import torch
import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pathlib
import pickle
import to_window, pretrained

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

def propose(candidates, old_1, old_2):
    while True:
        proposal = random.choice(candidates)
        if any(map(lambda x: x in proposal, old_1)):
            continue #restrict A: Jon Adams-Wick  B: John Wick
        if any(map(lambda x: x in proposal, old_2)):
            continue
        proposal = proposal.split()
        if any(map(lambda x: x.isdecimal(), proposal)):
            continue #there are nouns such as 'Academy 1885'
        return proposal

class DS_Augmented(Dataset):
    """Just return tokenized sentences(integer lists) and spans"""
    def __init__(self, tsv, tokenizer, replace_prob=.6, labels=True,
                 return_raw=False, return_ids=False):
        super().__init__()
        male_prns = set('he him his'.split())
        fmale_prns = set('her she'.split())

        df = pd.read_csv(tsv, '\t').set_index('ID')
        with open('./data/gap-coreference/nouns.pkl', 'rb') as f:
            data = pickle.load(f)
            male_dct = data['male_dct']
            fmale_dct = data['female_dct']

        self.tokenizer = tokenizer
        self.replace_prob = replace_prob
        self.df = df
        self.mdct = male_dct
        self.fdct = fmale_dct
        self.labels = labels
        self.return_raw = return_raw
        self.return_ids = return_ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        orig_row = self.df.iloc[i]
        splitted = (orig_row.A.split(), orig_row.B.split())
        orig_a, orig_b = splitted

        if self.replace_prob == 0:
            return self._process_row(orig_row)

        if not set(orig_a) & set(orig_b):
            for a in orig_a:
                if a in orig_row.B:
                    return self._process_row(orig_row)
            for b in orig_b:
                if b in orig_row.A:
                    return self._process_row(orig_row)

        if (random.random() > self.replace_prob or
                ',' in orig_row.A or ',' in orig_row.B or
                '(' in orig_row.A or '(' in orig_row.B or
                ')' in orig_row.A or ')' in orig_row.B or
                len(orig_a) >= 4 or len(orig_b) >= 4 or
                '**' in orig_row.A or '**' in orig_row.B or
                any(map(lambda x: x.lower() in 'he his him'.split(), orig_a)) or
                any(map(lambda x: x.lower() in 'she her'.split(), orig_a)) or
                any(map(lambda x: x.lower() in 'he his him'.split(), orig_b)) or
                any(map(lambda x: x.lower() in 'she her'.split(), orig_b))):
            return self._process_row(orig_row)

        new_row = orig_row.copy(deep=True)
        dct = self.mdct
        if orig_row.Pronoun.lower() in ('her', 'she'):
            dct = self.fdct

        sent_txt = orig_row.Text
        offsets = sorted((orig_row[f'{x}-offset'], x) for x in 'A B Pronoun'.split())
        sent_txts = [
            sent_txt[:offsets[0][0]],
            sent_txt[offsets[0][0]:offsets[1][0]],
            sent_txt[offsets[1][0]:offsets[2][0]],
            sent_txt[offsets[2][0]:],
        ]

        proposal_a = propose(dct[len(orig_a)], orig_a, orig_b)
        proposal_b = propose(dct[len(orig_b)], orig_b, orig_a)
        amap = dict(itertools.chain(zip(orig_a, proposal_a), zip(orig_b, proposal_b)))
        new_row['A'] = ' '.join(amap[x] for x in orig_a)
        new_row['B'] = ' '.join(amap[x] for x in orig_b)

        new_a, new_b = orig_row.A, orig_row.B

        for a, b in amap.items():
            a = re.escape(a)
            for i in range(4):
                if sent_txts[i].startswith('*'):
                    sent_txts[i] = re.sub(r'^'+a+r'(\W|\b)', b+r'\1', sent_txts[i])
                sent_txts[i] = re.sub(r'(\W|\b)'+a+r'(\W|\b)', r'\1'+b+r'\2', sent_txts[i])
        new_row.Text = ''.join(sent_txts)
        #set offsets
        l = len(sent_txts[0])
        new_row[f'{offsets[0][1]}-offset'] = l
        l += len(sent_txts[1])
        new_row[f'{offsets[1][1]}-offset'] = l
        l += len(sent_txts[2])
        new_row[f'{offsets[2][1]}-offset'] = l
        return self._process_row(new_row)

    def _process_row(self, row):
        if self.return_raw:
            return row.Text
        tokens, spans = pretrained.tokenize(row, self.tokenizer)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        spans = torch.tensor([s+1 for s in spans])

        if self.labels:
            if row['A-coref']:
                y = 0
            elif row['B-coref']:
                y = 1
            else:
                y = 2
            
            if self.return_ids:
                return tokens, spans, y,  row.name
            else:
                return tokens, spans, y
        elif self.return_ids:
            return tokens, spans, row.name
        return tokens, spans

    @staticmethod
    def collate_fn(inputs):
        sents = [inp[0] for inp in inputs]
        spans = [inp[1] for inp in inputs]
        outys = [inp[2] for inp in inputs]

        sents = torch.nn.utils.rnn.pad_sequence(sents, batch_first=True, padding_value=0)
        spans = torch.stack(spans)
        outys = torch.tensor(outys)
        return sents, spans, outys

class DS_v3(Dataset):
    """return means, but also distances and sizes of them"""
    def __init__(self, pkl, tsv_path, layer=-1):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=True)
        df = pd.read_csv(tsv_path, '\t')
        if isinstance(pkl, np.ndarray):
            data = pkl
            if layer >= 0:
                data = data[layer]
        else:
            with open(pkl, 'rb') as f:
                data = pickle.load(f)[layer]
        a_mat = torch.zeros((len(df), BERT_DIM), dtype=torch.float32)
        b_mat = torch.zeros_like(a_mat)
        p_mat = torch.zeros((len(df), BERT_DIM), dtype=torch.float32)
        self.y = [0]*len(df)

        for i, row in df.iterrows():
            a = torch.from_numpy(np.array(data[i][0]))
            b = torch.from_numpy(np.array(data[i][1]))
            c = torch.from_numpy(np.array(data[i][2]))
            a_mat[i] = a.mean(0)
            b_mat[i] = b.mean(0)
            p_mat[i] = c

            if row['A-coref']:
                self.y[i] = 0
            elif row['B-coref']:
                self.y[i] = 1
            else:
                self.y[i] = 2
        self.a_mat = a_mat.to(dev)
        self.b_mat = b_mat.to(dev)
        self.p_mat = p_mat.to(dev)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.a_mat[i], self.b_mat[i], self.p_mat[i],
                self.y[i])
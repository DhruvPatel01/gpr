import click
import numpy as np
import tqdm
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import pickle

def tokenize_row(row, tokenizer):
    txt = row.Text
    pronoun = row.Pronoun
    pr_offset = row['Pronoun-offset']
    ans = row.A if row['A-coref'] else row.B
    ans_offset = row['A-offset'] if row['A-coref'] else row['B-offset']
    
    if pr_offset < ans_offset:
        a = pr_offset
        atok = ' [PAD] '
        b = ans_offset
        btok = ' [SEP] '
    else:
        b = pr_offset
        btok = ' [PAD] '
        a = ans_offset
        atok = ' [SEP] '
    txt = '[CLS] ' + txt[:a] + atok + txt[a:b] + btok + txt[b:]
    
    toks = tokenizer.tokenize(txt)

    pr_index = toks.index('[PAD]')
    toks.pop(pr_index)
    ans_index = toks.index('[SEP]')
    toks.pop(ans_index)
    if pr_index > ans_index:
        pr_index -= 1
    ans = tokenizer.tokenize(ans)

    tok_ids= tokenizer.convert_tokens_to_ids(toks)
    txt = txt.replace('[SEP]', '')
    txt = txt.replace('[PAD]', '')
    obj = {
        "tokens": toks,
        "tok_ids": tok_ids,
        "pronoun": tokenizer.tokenize(pronoun),
        "pronoun_index": pr_index,
        "ans": ans,
        "ans_index": ans_index
    }
    return obj

def tokenize_tsv(inp):
    df = pd.read_csv(inp, sep='\t').set_index('ID')
    toret = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for id, row in tqdm.tqdm(df.iterrows(), "Processing", total=len(df)):
        try:
            obj = tokenize_row(row, tokenizer)
        except:
            print(id)
            raise
        obj['ID'] = id
        toret.append(obj)
    return toret

@click.command()
@click.argument('inp')
@click.argument('out')
def main(inp, out):
    ret = tokenize_tsv(inp)
    for obj in tqdm.tqdm(ret, "Saving"):
        with open(out + '/' + obj['ID'] + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

if __name__ == "__main__":
    main()
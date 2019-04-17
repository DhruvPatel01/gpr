import click
import numpy as np
import tqdm
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import pickle

def tokenize(row, tokenizer):
    break_points = sorted(
        [
            ("A", row["A-offset"], row["A"]),
            ("B", row["B-offset"], row["B"]),
            ("P", row["Pronoun-offset"], row["Pronoun"]),
        ], key=lambda x: x[0]
    )
    tokens, spans, current_pos = [], {}, 0
    for name, offset, text in break_points:
        tokens.extend(tokenizer.tokenize(row["Text"][current_pos:offset]))
        # Make sure we do not get it wrong
        assert row["Text"][offset:offset+len(text)] == text
        # Tokenize the target
        tmp_tokens = tokenizer.tokenize(row["Text"][offset:offset+len(text)])
        spans[name] = [len(tokens), len(tokens) + len(tmp_tokens)]
        tokens.extend(tmp_tokens)
        current_pos = offset + len(text)
    tokens.extend(tokenizer.tokenize(row["Text"][current_pos:offset]))
    assert spans["P"][0] == spans["P"][1]-1
    return tokens, (spans["A"] + spans["B"] + [spans["P"][0]])

def tokenize_tsv(inp, layers_to_save=[9, 10, 11]):
    df = pd.read_csv(inp, sep='\t')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(dev)

    answ = [[] for i in layers_to_save]
    with torch.no_grad():
        for id, row in tqdm.tqdm(df.iterrows(), "Processing", total=len(df)):
            sent, spans = tokenize(row, tokenizer)
            sent = ['[CLS]'] + sent + ['[SEP]']
            spans = [s+1 for s in spans]
            sent = tokenizer.convert_tokens_to_ids(sent)
            sent = torch.tensor([sent]).to(dev)
            encoding, _ = model(sent)
            for i, n in enumerate(layers_to_save):
                A = encoding[n][0][spans[0]:spans[1]].cpu().numpy().tolist()
                B = encoding[n][0][spans[2]:spans[3]].cpu().numpy().tolist()
                P = encoding[n][0][spans[4]].cpu().numpy().tolist()
                answ[i].append((A, B, P))
    return answ

@click.command()
@click.argument('inp')
@click.argument('out')
def main(inp, out):
    ret = tokenize_tsv(inp)
    with open(out, 'wb') as f:
        pickle.dump(ret, f)

if __name__ == "__main__":
    main()
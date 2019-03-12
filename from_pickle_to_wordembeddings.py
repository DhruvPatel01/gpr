import torch
import click
import glob
from pytorch_pretrained_bert import BertModel
import numpy as np
import tqdm
import pathlib
import pickle

@click.command()
@click.argument('inp')
@click.argument('out')
def main(inp, out):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = pathlib.Path(inp)
    objs = []
    for obj in path.glob('*.pkl'):
        with open(obj, 'rb') as f:
            objs.append(pickle.load(f))
    model = BertModel.from_pretrained('bert-base-uncased').to(dev)
    model.eval()
    with torch.no_grad():
        for obj in tqdm.tqdm(objs[:1]):
            ten = torch.tensor(obj['tok_ids']).view(1, -1).to(dev)
            sen = torch.zeros_like(ten).to(dev)
            embd, _ = model(ten, sen)
            ans = embd[-1][0].cpu().numpy()
            np.save(out+f'/{obj["ID"]}', ans)

if __name__ == "__main__":
    main()

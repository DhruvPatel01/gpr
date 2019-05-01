#Experiment script to get mean vectors, attended vectors and attentions
import pickle
import torch
import tqdm
from pytorch_pretrained_bert import BertModel, BertTokenizer

import dataset
import withaugmented_w_attn as waug

if __name__ == "__main__":
    dev = 'cuda'
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=False)
    bert = BertModel.from_pretrained('bert-large-cased').to(dev)
    attn = waug.MultiAttention(256, True).to('cuda')
    ds = dataset.DS_Augmented('./data/gap-coreference/gap-validation.tsv', tokenizer, 0)
    
    attn.load_state_dict(torch.load('./experiments/large-19-dpr-attn/best_attn.pt'))

    generated = []
    mean = []
    attentions = []
    for elem in tqdm.tqdm(ds):
        sent, spans, y = elem
        sent = sent[None].cuda()
        spans = spans[None].cuda()

        with torch.no_grad():
            embd, _ = bert(sent)
            embd = embd[19]
            a_mask, b_mask = waug.generate_masks(embd, spans, dev)
            p = embd[0, spans[0, 4]][None]
            a, b, aattn, battn = attn(embd, p, a_mask, b_mask)
        a, b = a[0], b[0]
        aattn, battn = aattn[0], battn[0]
        spans = spans[0]
        p, q, r, s, t = spans.cpu().numpy()
        embd = embd[0].cpu().numpy()
        mean.append(embd[p:q].mean(0))
        mean.append(embd[r:s].mean(0))
        generated.append(a.cpu().numpy())
        generated.append(b.cpu().numpy())
        attentions.append(aattn.cpu().numpy())
        attentions.append(battn.cpu().numpy())

    with open('./vectors.pkl', 'wb') as f:
        pickle.dump({
            'mean': mean,
            'attn': attentions,
            'generated': generated
        }, f)
import click
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

def get_correct_ans(row):
    y = 2
    if 'A-coref' in row and row['A-coref']:
        y = 0
    elif 'B-coref' in row and row['B-coref']:
        y = 1
    return y

def predict(row):
    p = -1
    for i, k in enumerate('A B Neither'.split()):
        if row[k+'-prob'] > p:
            p = row[k+'-prob']
            ans = i
    return ans

@click.command()
@click.option('--gender', '-g', default='')
@click.argument("tsv")
@click.argument("results")
def main(gender, tsv, results):
    df = pd.read_csv(tsv, '\t').set_index("ID")
    df['y'] = df.apply(get_correct_ans, 1)
    res = pd.read_csv(results).set_index("ID")
    res = res.rename(columns={'A': 'A-prob', 'B':'B-prob', "NEITHER":"Neither-prob"})
    res['predicted'] = res.apply(predict, 1)

    df = df.join(res)
    assert not any(df['A-prob'].isna())

    if gender.startswith('m'):
        prns = 'he his him'.split()
    elif gender.startswith('f'):
        prns = 'she her hers'.split()
    else:
        prns = 'he his him she her'.split()
    prns.extend([p.capitalize() for p in prns])

    idx = df['Pronoun'].str.lower().isin(prns)
    df = df[idx]

    print(classification_report(
        df['y'], df['predicted'],
        labels=[0,1,2], 
        target_names='A B Neither'.split(),
        digits=4
    ))

    #cross entropy
    loss = 0
    for i, row in df.iterrows():
        correct = 'A B Neither'.split()[row.y]
        prob = row[correct+'-prob']
        loss -= np.log(prob)
    loss /= len(df)
    print("Cross Entropy:", loss)

if __name__ == "__main__":
    main()
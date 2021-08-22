import pandas as pd

def gen_csv(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    df = pd.DataFrame(data, columns=['idx', 'text', 'label'])
    df = df.drop(['idx'], axis=1)
    df.loc[df['text'] == '', 'text'] = '.'
    df.to_csv(filename[:-4] + '.csv', index=None)

gen_csv("ratings_train.txt")
gen_csv("ratings_test.txt")
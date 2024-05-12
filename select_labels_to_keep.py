import pandas as pd


def select_labels(filepath, columns_to_drop, columns_to_keep):
    df = pd.read_csv(filepath)

    df.loc[(df['G'] == 1) | (df['A'] == 1) | (df['H'] == 1), 'O'] = 1

    df.drop(columns=['G', 'A', 'H'], inplace=True)

    df.to_csv(filepath, index=False)


columns_to_drop = ['G', 'A', 'H', 'O']
columns_to_check = ['N', 'D', 'C', 'M']

for i in ['data/train.csv', 'data/val.csv', 'data/test.csv']:
    select_labels(i, columns_to_drop, columns_to_check)

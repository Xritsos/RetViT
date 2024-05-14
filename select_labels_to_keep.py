import pandas as pd


def select_labels(filepath, columns_to_drop, columns_to_keep):
    df = pd.read_csv(filepath)

    df.drop(columns=columns_to_drop, inplace=True)

    mask = (df[columns_to_check] == 0).all(axis=1)

    df = df[~mask]

    # df.loc[(df['G'] == 1) | (df['A'] == 1) | (df['H'] == 1), 'O'] = 1
    # 
    # df.drop(columns=['G', 'A', 'H'], inplace=True)
    # 
    df.to_csv(filepath, index=False)


columns_to_drop = ['C', 'A', 'M', 'O']
columns_to_check = ['N', 'D', 'G', 'H']

for i in ['data/train.csv', 'data/val.csv', 'data/test.csv']:
    select_labels(i, columns_to_drop, columns_to_check)

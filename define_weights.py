import pandas as pd

df=pd.read_csv('data/train.csv')

print(df['N'].value_counts()[1]/len(df))
print(df['D'].value_counts()[1]/len(df))
print(df['G'].value_counts()[1]/len(df))
print(df['C'].value_counts()[1]/len(df))
print(df['A'].value_counts()[1]/len(df))
print(df['H'].value_counts()[1]/len(df))
print(df['M'].value_counts()[1]/len(df))
print(df['O'].value_counts()[1]/len(df))


print(f'Number of samples: {len(df)}')



import pandas as pd

df=pd.read_csv('data/train.csv')

print(round(len(df)/df['N'].value_counts()[1],2))
print(round(len(df)/df['D'].value_counts()[1],2))
print(round(len(df)/df['G'].value_counts()[1],2))
print(round(len(df)/df['C'].value_counts()[1],2))
print(round(len(df)/df['A'].value_counts()[1],2))
print(round(len(df)/df['H'].value_counts()[1],2))
print(round(len(df)/df['M'].value_counts()[1],2))
print(round(len(df)/df['O'].value_counts()[1],2))


print(f'Number of samples: {len(df)}')



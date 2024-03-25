import pandas as pd

<<<<<<< Updated upstream
df = pd.read_excel('./data/ODIR-5K.xlsx')
=======
df = pd.read_csv('../data/filtered_data.csv')
>>>>>>> Stashed changes

print(df.head())


def print_label_cases():
    print('Age-related macular degeneration: ', df['A'].value_counts()[1])
    print('Cataract: ', df['C'].value_counts()[1])
    print('Diabetes Retinopathy: ', df['D'].value_counts()[1])
    print('Glaukoma: ', df['G'].value_counts()[1])
    print('Hypertension: ', df['H'].value_counts()[1])
    print('Myopia: ', df['M'].value_counts()[1])
    print('Other Abnormalities: ', df['O'].value_counts()[1])
    print('Normal: ', df['N'].value_counts()[1])


print_label_cases()

df = df.sort_values(by='ID')
df.to_csv('../data/filtered_data.csv', index=False)
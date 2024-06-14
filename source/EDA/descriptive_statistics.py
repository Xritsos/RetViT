import matplotlib.pyplot as plt
import pandas as pd

merged_df = pd.read_csv('./data/final_dataset.csv')

# plot class distribution
class_counts = merged_df[['N', 'D', 'C', 'M']].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

label_corr = merged_df[['N', 'D', 'C', 'M']].corr()
print("Label Correlation:")
print(label_corr)


label_cardinality = merged_df[['N', 'D', 'C', 'M']].sum(axis=1).mean()
print("Label Cardinality:", label_cardinality)

total_labels = merged_df[['N', 'D', 'C', 'M']].sum().sum()
unique_labels = merged_df[['N', 'D', 'C', 'M']].any(axis=1).sum()
label_density = unique_labels / total_labels
print("Label Density:", label_density)

label_frequency = merged_df[['N', 'D', 'C', 'M']].mean()
print("Label Frequency:")
print(label_frequency)

import seaborn as sns

# Plot heatmap of label correlation
plt.figure(figsize=(8, 6))
sns.heatmap(label_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Label Correlation')
plt.show()

label_cardinality = merged_df[['N', 'D', 'C', 'M']].sum(axis=1)

# Plot bar plot of label cardinality
plt.figure(figsize=(8, 6))
label_cardinality.value_counts().sort_index().plot(kind='bar', color='orange')
plt.title('Label Cardinality')
plt.xlabel('Number of Labels per Instance')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()
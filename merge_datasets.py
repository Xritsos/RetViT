import pandas as pd

# Load train, validation, and test datasets
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

# Concatenate the datasets
merged_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Display the first few rows of the merged dataframe
print(merged_df.head())

# Save the merged dataframe to a new CSV file if needed
merged_df.to_csv("data/final_dataset.csv", index=False)
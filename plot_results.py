import pandas as pd
import matplotlib.pyplot as plt


metrics_df = pd.read_csv('logs/experiment_name/version_5/metrics.csv')

fig, axs = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    label_train_f1 = metrics_df.groupby('epoch')[f'{i} train f1'].mean()
    label_val_f1 = metrics_df.groupby('epoch')[f'{i} val f1'].mean()
    epochs = range(1, len(label_train_f1) + 1)

    row = i // 4
    col = i % 4

    axs[row, col].plot(epochs, label_train_f1, 'b', label=f'Mean Training F1 for label {i}')
    axs[row, col].plot(epochs, label_val_f1, 'r', label=f'Mean Validation F1 for label {i}')
    axs[row, col].set_title(f'Train - Val F1 for label {i}')
    axs[row, col].set_xlabel('Epochs')
    axs[row, col].set_ylabel('F1 Score')
    axs[row, col].legend()
    axs[row, col].set_ylim(0, 1)

plt.tight_layout()
plt.show()


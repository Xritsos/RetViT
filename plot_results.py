import pandas as pd
import matplotlib.pyplot as plt
import config

metrics_df = pd.read_csv('logs/experiment_name/swin_default/metrics.csv')

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

for i in range(config.num_labels):
    label_train_f1 = metrics_df.groupby('epoch')[f'{i} train f1'].mean()
    label_val_f1 = metrics_df.groupby('epoch')[f'{i} val f1'].mean()
    epochs = range(1, len(label_train_f1) + 1)

    col = i

    axs[col].plot(epochs, label_train_f1, 'b', label=f'Mean Training F1 for label {i}')
    axs[col].plot(epochs, label_val_f1, 'r', label=f'Mean Validation F1 for label {i}')
    axs[col].set_title(f'Train - Val F1 for label {i}')
    axs[col].set_xlabel('Epochs')
    axs[col].set_ylabel('F1 Score')
    axs[col].legend()
    axs[col].set_ylim(0, 1)

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

metrics_df = pd.read_csv('lightning_logs/version_2/metrics.csv')

mean_train_loss = metrics_df.groupby('epoch')['training_loss_epoch'].mean()
mean_val_loss = metrics_df.groupby('epoch')['validation_loss'].mean()
epochs = range(1, len(mean_train_loss) + 1)

mean_train_f1 = metrics_df.groupby('epoch')['training F1 Score_epoch'].mean()
mean_val_f1 = metrics_df.groupby('epoch')['validation F1 Score'].mean()

fig, axs = plt.subplots(2)

axs[0].plot(epochs, mean_train_loss, 'b', label='Mean Training Loss')
axs[0].plot(epochs, mean_val_loss, 'r', label='Mean Validation Loss')
axs[0].set_title('Mean Training and Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(epochs, mean_train_f1, 'b', label='Mean Training F1 Score')
axs[1].plot(epochs, mean_val_f1, 'r', label='Mean Validation F1 Score')
axs[1].set_title('Mean Training and Validation F1 Score')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('F1 Score')
axs[1].legend()

plt.tight_layout()
plt.show()
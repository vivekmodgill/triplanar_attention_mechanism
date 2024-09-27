import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
metrics_df = pd.read_csv('/Users/vivek/Documents/lesionsegmentation/model_alpha0.7/metrics_03.csv')

# To smooth the lines, we can apply a rolling average (with a window of 5 for example)
metrics_df['training_loss_smooth']   = metrics_df['training/loss'].rolling(window=5, min_periods=1).mean()
metrics_df['validation_loss_smooth'] = metrics_df['validation/loss'].rolling(window=5, min_periods=1).mean()
metrics_df['training_bce_smooth']    = metrics_df['training/bce'].rolling(window=5, min_periods=1).mean()
metrics_df['validation_bce_smooth']  = metrics_df['validation/bce'].rolling(window=5, min_periods=1).mean()
metrics_df['training_bsd_smooth']    = metrics_df['training/bsd'].rolling(window=5, min_periods=1).mean()
metrics_df['validation_bsd_smooth']  = metrics_df['validation/bsd'].rolling(window=5, min_periods=1).mean()

# Create a combined plot with subplots
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# Training and Validation Loss
ax[0, 0].plot(metrics_df['epoch'], metrics_df['training_loss_smooth'], label='Training Loss', color='b', linestyle='-')
ax[0, 0].plot(metrics_df['epoch'], metrics_df['validation_loss_smooth'], label='Validation Loss', color='b', linestyle='--')
ax[0, 0].set_xlabel('Epoch')
ax[0, 0].set_ylabel('Loss')
ax[0, 0].set_title('Training and Validation Loss')
ax[0, 0].legend()
ax[0, 0].grid(True)

# Training and Validation BCE
ax[0, 1].plot(metrics_df['epoch'], metrics_df['training_bce_smooth'], label='Training BCE', color='r', linestyle='-')
ax[0, 1].plot(metrics_df['epoch'], metrics_df['validation_bce_smooth'], label='Validation BCE', color='r', linestyle='--')
ax[0, 1].set_xlabel('Epoch')
ax[0, 1].set_ylabel('BCE')
ax[0, 1].set_title('Training and Validation BCE')
ax[0, 1].legend()
ax[0, 1].grid(True)

# Training and Validation BSD
ax[1, 0].plot(metrics_df['epoch'], metrics_df['training_bsd_smooth'], label='Training BSD', color='g', linestyle='-')
ax[1, 0].plot(metrics_df['epoch'], metrics_df['validation_bsd_smooth'], label='Validation BSD', color='g', linestyle='--')
ax[1, 0].set_xlabel('Epoch')
ax[1, 0].set_ylabel('BSD')
ax[1, 0].set_title('Training and Validation BSD')
ax[1, 0].legend()
ax[1, 0].grid(True)

# Empty subplot or additional metric
ax[1, 1].axis('off')

plt.tight_layout()
plt.savefig('training_validation_metrics03.png', dpi=300)
plt.show()

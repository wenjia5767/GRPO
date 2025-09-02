import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('training_loss.csv')

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(data['step'], data['loss'], marker='.', linestyle='-', markersize=4)

# Add titles and labels for clarity
plt.title('DPO Training Loss per Step')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.grid(True)

# Save the plot to a file
plt.savefig('dpo_training_loss_curve.png')

print("Plot saved as dpo_training_loss_curve.png")

# Optionally, display the plot
plt.show()
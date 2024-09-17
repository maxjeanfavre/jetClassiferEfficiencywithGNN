import re
import matplotlib.pyplot as plt
import os


current_file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_file_path)


# Open the log file
with open(os.path.join(directory_path, "train.log")) as file:
    log_contents = file.read()

# Define regular expressions for extracting the epoch, training loss, and validation loss
epoch_pattern = r"epoch: (\d+)"
train_loss_pattern = r"epoch train loss: ([\d.]+)"
validation_loss_pattern = r"epoch validation loss: ([\d.]+)"

# Initialize lists to store extracted values
epochs = []
train_losses = []
validation_losses = []

# Iterate over each line of the log contents
for line in log_contents.split("\n"):
    # Extract epoch
    epoch_match = re.search(epoch_pattern, line)
    if epoch_match:
        epochs.append(int(epoch_match.group(1)))
    
    # Extract train loss
    train_loss_match = re.search(train_loss_pattern, line)
    if train_loss_match:
        train_losses.append(float(train_loss_match.group(1)))
    
    # Extract validation loss
    validation_loss_match = re.search(validation_loss_pattern, line)
    if validation_loss_match:
        validation_losses.append(float(validation_loss_match.group(1)))

# Print extracted values
#print("Epochs:", epochs)
#print("Training Losses:", train_losses)
#print("Validation Losses:", validation_losses)


# Plot training and validation losses
plt.plot(epochs, train_losses, label='Training Loss', color='r')
plt.plot(epochs, validation_losses, label='Validation Loss', color='g')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid()

# Show plot
#plt.show()

# Save plot
plt.savefig(os.path.join(directory_path, "training_validation_losses.pdf"))

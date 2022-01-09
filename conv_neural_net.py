import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from mnist_dataset import MnistTest, MnistTrain
from conv_autoencoder import Conv_Autoencoder
from neural_net_train import train
from neural_net_test import test

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import data
train_data = MnistTrain()
test_data = MnistTest()

# Data needs to be modified to 3D arrays
train_data.x = train_data.x.reshape(train_data.__len__(), 1, 28, 28)
test_data.x = test_data.x.reshape(test_data.__len__(), 1, 28, 28)

# Initialize variables
learning_rate = 0.01
batch = 1000
n_epochs = 20


# Create the model and pass it to the device
model = Conv_Autoencoder().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Create dataloaders
train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True)


# Variables that store the training and the test losses
test_outputs = []
train_outputs = []

# Iterate for every epoch
for epoch in range(n_epochs):
    train_outputs.append(train(epoch, train_loader, model, criterion, optimizer))
    test_outputs.append(test(test_loader, model, criterion))


# Extract losses from outputs
train_loss = []
test_loss = []

for loss in train_outputs:
    train_loss.append(loss[0])
for loss in test_outputs:
    test_loss.append(loss[0])

# Plot losses
plot1 = plt.figure(1)
plt.plot(train_loss, "-o")
plt.plot(test_loss, "-o")
plt.xlabel("epoch")
plt.ylabel("RMSE")
plt.legend(["Train", "Test"])
plt.title("Train vs Test Losses")

for epoch in range(0, n_epochs, 6):
    plt.figure(figsize=(9, 2))
    imgs = train_outputs[epoch][1].detach().cpu().numpy()
    recogn = train_outputs[epoch][2].detach().cpu().numpy()
    for i, item in enumerate(imgs):
        # Plot only the first 10 images
        if i >= 9:
            break
        plt.subplot(2, 9, i + 1)
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0], cmap="gray")

    for i, item in enumerate(recogn):
        # Plot only the first 10 images
        if i >= 9:
            break
        plt.subplot(2, 9, 9 + i + 1)
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0], cmap="gray")


plt.show()

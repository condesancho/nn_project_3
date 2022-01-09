import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import time
import matplotlib.pyplot as plt

from mnist_dataset import MnistTest, MnistTrain
from autoencoder import Autoencoder
from neural_net_train import train
from neural_net_test import test

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


# Import data as Dataset
train_data = MnistTrain()
test_data = MnistTest()


# Initialize variables
learning_rate = 1e-3
batch = 1000
n_epochs = 20
hidden_layers = [254, 128, 64]
middle_layer = 9

# Create dataloaders
train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True)


exec_times = []
total_train_loss = []
total_test_loss = []
legend = []

learning = [0.0001, 0.001, 0.005, 0.01]
for learningrate in learning:

    # Variables that store the training and the test losses
    test_outputs = []
    train_outputs = []

    train_loss = []
    test_loss = []

    model = Autoencoder(hidden_layers=hidden_layers, middle_layer=middle_layer).to(
        device
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    # Start timer
    start_time = time.time()

    # Iterate for every epoch
    for epoch in range(n_epochs):
        train_outputs.append(train(epoch, train_loader, model, criterion, optimizer))
        test_outputs.append(test(test_loader, model, criterion))

    exec_times.append(time.time() - start_time)

    for loss in train_outputs:
        train_loss.append(loss[0])
    for loss in test_outputs:
        test_loss.append(loss[0])

    # Store the final loss of the train and test set
    total_test_loss.append(train_loss)
    total_train_loss.append(test_loss)

    legend.append(f"learning rate = {learningrate}")
    print(
        "The time that took the model to train is %s seconds."
        % (time.time() - start_time)
    )

total_test_loss = np.array(total_test_loss)
total_train_loss = np.array(total_train_loss)

# Plot losses
plot1 = plt.figure(1)
plt.plot(total_test_loss.T)
plt.xlabel("epoch")
plt.ylabel("RMSE")
plt.legend(legend)
plt.title("Train losses for different learning rates")

# Plot accuracies
plot2 = plt.figure(2)
plt.plot(total_train_loss.T)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(legend)
plt.title("Test losses for different learning rates")

plot3 = plt.figure(3)
plt.plot(range(len(learning)), exec_times, "-o")
plt.xticks(range(len(learning)), learning)
plt.xlabel("learning rate")
plt.ylabel("execution times")


plt.show()

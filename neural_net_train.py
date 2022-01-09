import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training function
def train(epoch, train_loader, model, criterion, optimizer):

    temp_loss = 0

    for (images, _) in train_loader:
        # Pass data to the device
        images = images.to(device)

        # Forward
        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        temp_loss += loss.item()

    # Store the mean loss of the epoch
    train_loss = math.sqrt(temp_loss / len(train_loader))

    # Print the values
    print(f"Epoch: {epoch+1}, Loss: {train_loss:.4f}")

    return train_loss, images, reconstructed

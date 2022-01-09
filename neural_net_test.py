import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing function
def test(test_loader, model, criterion):
    with torch.no_grad():
        temp_loss = 0

        for (images, _) in test_loader:
            images = images.to(device)

            reconstructed = model(images)

            loss = criterion(reconstructed, images)

            # Accumulate loss
            temp_loss += loss.item()

        # Store the mean loss of the epoch
        test_loss = math.sqrt(temp_loss / len(test_loader))

        # Print the values
        print(f"Test Loss: {test_loss:.4f}\n")

        return test_loss, images, reconstructed

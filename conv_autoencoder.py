import torch.nn as nn

# Define the NN
class Conv_Autoencoder(nn.Module):
    # Init function
    def __init__(self):
        super(Conv_Autoencoder, self).__init__()
        # The size of the image stays the same with 8 output channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (1, 28, 28) -> (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (16, 14, 14) -> (32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),  # (32, 7, 7) -> (64, 1, 1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            # If output_padding=0 then output would be (16, 13, 13)
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    # Forward function
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

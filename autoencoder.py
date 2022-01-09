import torch.nn as nn

# Define the NN
class Autoencoder(nn.Module):
    # Init function
    def __init__(self, hidden_layers=[254, 128], middle_layer=64, activation="relu"):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential()

        # Make a dict for the activation functions
        self.activations = nn.ModuleDict({"relu": nn.ReLU(), "sigmoid": nn.Sigmoid()})

        # The input on the first layer is the number of columns of the dataset
        input_size = 28 * 28

        # Iterate for the appropriate number of layers
        for i, layer in enumerate(hidden_layers):
            self.encoder.add_module(f"hidden{i+1}", nn.Linear(input_size, layer))

            # Pass the activation function according to the activation variable
            self.encoder.add_module(f"activation", self.activations[activation])

            # Change the input of the next layer
            input_size = layer

        # Add the middle layer of the autoencoder
        self.encoder.add_module(f"middle", nn.Linear(hidden_layers[-1], middle_layer))

        # Decoder
        self.decoder = nn.Sequential()

        input_size = middle_layer

        for i, layer in enumerate(reversed(hidden_layers)):
            self.decoder.add_module(
                f"hidden{len(hidden_layers)+i+1}", nn.Linear(input_size, layer)
            )

            # Pass the activation function according to the activation variable
            self.decoder.add_module(f"activation", self.activations[activation])

            # Change the input of the next layer
            input_size = layer

        self.decoder.add_module(f"output", nn.Linear(hidden_layers[0], 28 * 28))

        # The last layer of the decoder must pass through the sigmoid activation function
        self.decoder.add_module(f"activation", self.activations["sigmoid"])

    # Forward function
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

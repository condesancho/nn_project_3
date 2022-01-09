from mnist import MNIST
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import time

# Import data from the folder
mndata = MNIST("./samples")

xtrain, ytrain = mndata.load_training()
xtest, ytest = mndata.load_testing()

# Turn list to array
xtrain = np.array(xtrain)
xtest = np.array(xtest)

# Initialize the scalers
train_scaler = StandardScaler()
test_scaler = StandardScaler()

# Fit the data to the scalers
train_scaler = train_scaler.fit(xtrain)
test_scaler = test_scaler.fit(xtest)

# Transform the data
train = train_scaler.transform(xtrain)
test = test_scaler.transform(xtest)

n_components = [2, 9, 100, 0.9, 0.95]

plt.figure(figsize=(5, len(n_components)))
plt.title("Autoencoded images for different number of components")
images = xtrain[:5, :].reshape(-1, 28, 28)
for j in range(5):
    plt.subplot(len(n_components) + 1, 5, j + 1)
    plt.imshow(images[j, :, :], cmap="gray")
    plt.axis("off")

for i, components in enumerate(n_components):
    # Start timer
    start_time = time.time()

    # Initialize the PCA model
    pca = PCA(n_components=components)

    pca = pca.fit(train)

    # Encode
    pca_transformed_train = pca.transform(train)
    pca_transformed_test = pca.transform(test)

    # Decode
    decoded_train = pca.inverse_transform(pca_transformed_train)
    decoded_test = pca.inverse_transform(pca_transformed_test)

    decoded_train = train_scaler.inverse_transform(decoded_train)
    decoded_test = test_scaler.inverse_transform(decoded_test)

    train_rmse = mean_squared_error(xtrain, decoded_train, squared=False)
    test_rmse = mean_squared_error(xtest, decoded_test, squared=False)

    print(f"For {pca.n_components_} components")
    print(f"The RMSE of the train set is: {train_rmse}")
    print(f"The RMSE of the test set is: {test_rmse}")
    print("Time passed: %s seconds." % (time.time() - start_time))

    reconstructed = decoded_train[:5, :].reshape(-1, 28, 28)

    for j in range(5):
        plt.subplot(len(n_components) + 1, 5, (i + 1) * 5 + j + 1)
        plt.imshow(reconstructed[j, :, :], cmap="gray")
        plt.axis("off")

plt.show()

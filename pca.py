from mnist import MNIST
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# Import data from the folder
mndata = MNIST("../samples")

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

# Initialize the PCA model
pca = PCA(n_components=300)

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

print(f"The RMSE of the train set is: {train_rmse}")
print(f"The RMSE of the test set is: {test_rmse}")

first_img = xtrain[0, :]
first_img = first_img.reshape(28, 28)

decoded_first_image = decoded_train[0, :]
decoded_first_image = decoded_first_image.reshape(28, 28)

plot1 = plt.figure(1)
plt.imshow(first_img, cmap="gray")
plot2 = plt.figure(2)
plt.imshow(decoded_first_image, cmap="gray")
plt.show()

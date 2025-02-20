#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
os.getcwd()

# %% transform and load data
# TODO: set up image transforms
transform = transforms.Compose(
    [transforms.Resize((50,50)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

batch_size = 4

# TODO: set up train and test datasets
trainset = torchvision.datasets.ImageFolder(root='train', transform=transform)
testset = torchvision.datasets.ImageFolder(root='test', transform=transform)

# TODO: set up data loaders
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

# %%
CLASSES = ['affenpinscher', 'akita', 'corgi']
NUM_CLASSES = len(CLASSES)

# TODO: set up model class
class ImageMulticlassClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 1 because images are grayscale
        self.conv1 = nn.Conv2d(1, 6, 3) # out: (BS, 6, 30, 30)
        self.pool = nn.MaxPool2d(2, 2) # out: (BS, 6, 15, 15)
        self.conv2 = nn.Conv2d(6, 16, 3) # out: (BS, 16, 13, 13)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 11 * 11, 128) # out: (BS, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
# input = torch.rand(1, 1, 50, 50) # BS, C, H, W
model = ImageMulticlassClassificationNet()      
# model(input).shape

# %% loss function and optimizer
# TODO: set up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
# %% training
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # TODO: define training loop
        # zero gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs) # model(inputs)
        loss = loss_fn(outputs, labels)
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()

    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')

# %% test
y_test = []
y_test_hat = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f'Accuracy: {acc*100:.2f} %')
# %% confusion matrix
confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
# %%
def visualize_mislabeled_examples(testloader, y_test, y_test_hat, num_images=5):
    """
    Visualizes mislabeled examples from the test set.

    Parameters:
    - testloader: DataLoader for the test dataset.
    - y_test: List of true labels for the test set (class indices).
    - y_test_hat: List of predicted class indices for the test set.
    - num_images: Number of mislabeled images to display.
    """
    # Convert y_test and y_test_hat to numpy arrays for comparison
    y_test = np.array(y_test)
    y_test_hat = np.array(y_test_hat)

    # Find indices of mislabeled examples
    mismatches = np.where(y_test != np.argmax(y_test_hat, axis=1))[0]

    # Limit the number of images to display
    num_images = min(num_images, len(mismatches))

    # Set up the plot
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        idx = mismatches[i]
        # Get the corresponding image from the testloader
        img, _ = testloader.dataset[idx]

        # Create a subplot for each mismatched example
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')  # Squeeze to remove channel dimension
        plt.title(f'True: {y_test[idx]}\nPred: {y_test_hat[idx]}')
        plt.axis('off')

    plt.show()

# Example usage
visualize_mislabeled_examples(testloader, y_test, y_test_hat)
# %%
visualize_mislabeled_examples(trainloader, y_test, y_test_hat)

# %%
import matplotlib.pyplot as plt

# Assuming you have a DataLoader called 'train_loader'

for images, labels in trainloader:
    # Get the first image in the batch
    # First 0 is first image and 0 is the first (and only) channel
    image = images[0][0]

    print(images.shape)

    # Convert the image from tensor to numpy array
    image = image.numpy()

    # Normalize the image if it is not already in the range [0, 1]
    image = image / 255.0

    # Display the image
    plt.imshow(image)
    plt.show()

    # Break out of the loop after displaying the first image
    break

# %%
# first index is batch index
# second index is image index
# third index is channel index
plt.imshow((next(iter(trainloader)))[0][0][0].numpy())

# %%

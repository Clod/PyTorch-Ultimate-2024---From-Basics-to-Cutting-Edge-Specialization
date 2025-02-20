#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
# The data set consists of 50 samples from each of three species 
# of Iris (Iris setosa, Iris virginica and Iris versicolor). 
# Four features were measured from each sample: the length and 
# the width of the sepals and petals, in centimeters.
# %% data import
iris = load_iris()
X = iris.data
Y = iris.target

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# %% convert to float32

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# %% dataset
class IrisData(Dataset):
    def __init__(self, X, y):
        super().__init__()
        # Convert X and y to PyTorch tensors
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        # Convert y to LongTensor
        self.y = self.y.type(torch.LongTensor)
        # Get the length of the dataset
        self.len = self.X.shape[0]
    def __len__(self):
        # Return the length of the dataset
        return self.len
    def __getitem__(self, idx):
        # Return the item at index idx
        return self.X[idx], self.y[idx]

# %% dataloader
iris_data = IrisData(X_train, y_train)
train_loader = DataLoader(dataset=iris_data, batch_size=32, shuffle=True)

# %% check dims
print(f"X Shape: {iris_data.X.shape}, y shape: {iris_data.y.shape}")

# %% define class
class MultiClassNet(nn.Module):
    #% define layers lin1, lin2, log_softmax
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        # define layers
        # lin1: input layer (NUM_FEATURES) -> hidden layer (HIDDEN_FEATURES)
        self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        # lin2: hidden layer (HIDDEN_FEATURES) -> output layer (NUM_CLASSES)
        self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        # Output of the previous layer is a tensor of shape (batch_size, num_classes)
        # the softmax operation is performed along the second dimension, 
        # resulting in a tensor of shape (batch_size, num_classes) where 
        # each row represents the probabilities of each class.
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #% forward pass
        # pass the input through the first layer
        x = self.lin1(x)
        # apply the sigmoid activation function to the output of the first layer
        x = torch.sigmoid(x)
        # pass the output of the first layer through the second layer
        x = self.lin2(x)
        # apply the softmax activation function to the output of the second layer
        x = self.log_softmax(x)
        return x
    
# %% hyper parameters
NUM_FEATURES = iris_data.X.shape[1] # shape[1] is the number of features
HIDDEN = 6
NUM_CLASSES = len(iris_data.y.unique()) # 3 in Iris dataset case
# %% create model instance
model = MultiClassNet(NUM_FEATURES=NUM_FEATURES,NUM_CLASSES=NUM_CLASSES,HIDDEN_FEATURES=HIDDEN)
# %% loss function
criterion = nn.CrossEntropyLoss()
# %% optimizer
LR = 0.1
optimizer=torch.optim.SGD(model.parameters(), lr=LR)
# %% training
NUM_EPOCHS = 100

losses = []

# loop through the entire dataset
for epoch in range(NUM_EPOCHS):
    # X, y = batch
    for X, y in train_loader:
        optimizer.zero_grad()
        # forward pass
        y_hat_log = model(X)
        # calculate loss
        loss = criterion(y_hat_log, y)
        # calculate gradients (backpropagation)
        loss.backward()
        # update weights
        optimizer.step()

    losses.append(float(loss.data.detach().numpy()))
            
# %% show losses over epochs
sns.lineplot(x=range(NUM_EPOCHS), y=losses)


# %% test the model
X_test_torch = torch.from_numpy(X_test)

with torch.no_grad():
    y_test_hat_softmax = model(X_test_torch)
    y_test_hat = torch.max(y_test_hat_softmax.data, 1)

# %% Accuracy
accuracy_score(y_test, y_test_hat.indices)

# %%
from collections import Counter
Counter(y_test).most_common()[0]
# %%
most_common_cnt = Counter(y_test).most_common()[0][1]

print(f"Naive classifier accuracy: {most_common_cnt/len(y_test) * 100}) %")
# %%

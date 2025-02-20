#%% packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns
import numpy as np
from collections import Counter

print("Caca")
# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size = 0.2)

# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TODO: create instance of dataset
multilabel_train_data = MultilabelDataset(X_train, y_train)
multilabel_test_data = MultilabelDataset(X_test, y_test)

# TODO: create train loader
train_loader = DataLoader(dataset = multilabel_train_data, batch_size=32,shuffle=True)
test_loader = DataLoader(dataset = multilabel_test_data, batch_size=32, shuffle=True)
# %% model
# TODO: set up model class
# topology: fc1, relu, fc2
# final activation function??

# TODO: define input and output dim
# input_dim = ??
# output_dim = ??



# TODO: create a model instance
class MultilabelNetwork (nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MultilabelNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
input_dim = multilabel_train_data.X.shape[1]
output_dim = multilabel_train_data.y.shape[1]

model = MultilabelNetwork(input_size=input_dim, hidden_size=20, output_size=output_dim)
model.train()

# %% loss function, optimizer, training loop
# TODO: set up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
slope, bias = [], []
number_epochs = 100

# TODO: implement training loop
for epoch in range(number_epochs):
    for j, data in enumerate(train_loader):
        
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_hat = model(data[0])

        # compute loss
        loss = loss_fn(y_hat, data[1])
        
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

    # TODO: print epoch and loss at end of every 10th epoch
    if (epoch % 10 == 0):
        print(f"Epoch {epoch}, Loss: {loss.data}")
        losses.append(loss.item())
        
# %% losses
# TODO: plot losses
sns.scatterplot(x = range(len(losses)), y = losses)

# %% test the model
# TODO: predict on test set

# With model in evaluation mode
with torch.no_grad():
    y_test_pred = model(X_test).round()  

#%% Naive classifier accuracy
# TODO: convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'
y_test_str = [ str(i) for i in y_test.detach().numpy() ]
most_common_cnt = Counter(y_test_str).most_common()[0][1]

print(f"Naive accuracy: {most_common_cnt / len(y_test ) * 100}%")


# %% Test accuracy
# TODO: get test set accuracy
accuracy_score(y_test, y_test_pred)
# %%

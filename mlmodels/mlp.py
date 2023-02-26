import math
import torch
import gpytorch
import numpy as np
torch.set_default_dtype(torch.float64)


class MLP(torch.nn.Module):
    def __init__(self, nfeatures, ntargets, depth, width):
        """
        nfeatures: int,
            number of features
        ntargets: int,
            number of targets
        depth: int,
            number of layers
        width: int,
            number of nodes in each layer
        """
        super(MLP, self).__init__()
        self.nfeatures = nfeatures
        self.ntargets = ntargets
        self.layers = torch.nn.ModuleList([])
        self.relu = torch.nn.functional.relu
        first_layer = torch.nn.Linear(nfeatures, width)
        self.layers.append(first_layer)
        if depth > 2:
            for i in range(1, depth-1):
                self.layers.append(torch.nn.Linear(width, width))
            self.layers.append(torch.nn.Linear(width, ntargets))

        else:
            self.layers.append(torch.nn.Linear(width, ntargets))

        #print(f"Initiated MLP with {depth} layers consisting of {width} nodes.")

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = self.relu(l(x))
        return x

def train_and_test(x_train, x_test, y_train, y_test, epochs, model, criterion, optimizer, scheduler):
    training_error = np.zeros(epochs)
    testing_error = np.zeros(epochs)
    count = 0
    for i in range(epochs):
        model.train()
        output = model(x_train)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_error[i] = loss.item()
        model.eval()
        with torch.no_grad():
            preds = model(x_test)
            loss = criterion(preds, y_test)
            testing_error[i] = loss.item()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Loss: {:.4f}'
                       .format(i+1, epochs, loss.item()))

        if (i+1) % 500 == 0:
            scheduler.step()

    return training_error, testing_error, preds, output


if __name__ == "__main__":
    layers = [2, 5, 6, 7]
    layers2 = [1, 2]
    model1 = MLP(10, 1, 5, 10)
    model2 = MLP(10, 1, 5, 10)

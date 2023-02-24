import math
import torch
import gpytorch
import numpy as np
torch.set_default_dtype(torch.float64)


class MLP(torch.nn.Module):
    def __init__(self, nfeatures, ntargets, layers):
        """
        nfeatures: int,
            number of features
        ntargets: int,
            number of targets
        layers: list of ints,
            len(layers) - number of layers
            layer[i] - number of nodes in layer i
        """
        super(MLP, self).__init__()
        nlayers = len(layers)
        self.nfeatures = nfeatures
        self.ntargets = ntargets
        self.layers = torch.nn.ModuleList([])
        self.relu = torch.nn.functional.relu
        first_layer = torch.nn.Linear(nfeatures, layers[0])
        self.layers.append(first_layer)
        if nlayers > 2:
            for i in range(1, nlayers-1):
                self.layers.append(torch.nn.Linear(layers[i-1], layers[i]))
            self.layers.append(torch.nn.Linear(layers[-1], ntargets))

        else:
            self.layers.append(torch.nn.Linear(layers[1], ntargets))

        print(f"Initiated MLP with {nlayers} layers:")
        for i in range(nlayers):
            print(f"Layer {i+1}: {layers[i]} nodes.")

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            x = self.relu(x)
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
    model1 = MLP(10, 1, layers)
    model2 = MLP(10, 1, layers2)


import torch
import numpy as np
import matplotlib.pyplot as plt

# load mnist data
from torchvision import datasets, transforms
import torch.nn.functional as F

def load_data(batch_size=64):
    """
    Loads the MNIST data set and normalizes it to have mean 0 and standard deviation 1.
    """
    # transform data to tensor
    transform = transforms.ToTensor()
    # load data
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # create dataloader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


# load data
train_loader, test_loader = load_data()


# define model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# define loss function and optimizer
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()

def train(model, loss_fn, optimizer, train_loader, epochs=1):
    """
    Trains the model for one epoch.
    """
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(torch.float), target.to(torch.long)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    return model



def test(model, loss_fn, test_loader):
    """
    Tests the model on the test data.
    """
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(torch.float), target.to(torch.long)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct



def main():
    """
    Trains the model and tests it on the test data.
    """
    model = Model()
    train_loader, test_loader = load_data()
    model = train(model, loss_fn, optimizer, train_loader, epochs=1)
    test_loss, correct = test(model, loss_fn, test_loader)


if __name__ == '__main__':
    main()
    print('done')
    plt.show()
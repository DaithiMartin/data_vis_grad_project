
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch import optim
import numpy as np
import simplejson as json

class CNN(nn.Module):

    """
    It appears to be good practice to leave the last layer of the linear layers raw.
    Then to use nn.CrossEntropyLoss() which combines nn.LogSoftmax() and nn.NLLLoss().
    If probabilities are needed then you can nn.functional.softmax() the output of the NN.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # input = 1x28x28
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1), # 10x26x26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 10x13x13
        )

        self.lin_1 = nn.Linear(in_features=784, out_features=20)
        self.lin_2 = nn.Linear(in_features=20, out_features=15)
        self.lin_3 = nn.Linear(in_features=15, out_features=10)
        self.lin_4 = nn.Linear(in_features=10, out_features=10)


    def forward(self, x):
        x1 = self.conv(x)
        x2 = x.view(x1.size(0), -1)
        x3 = F.relu(self.lin_1(x2))
        x4 = F.relu(self.lin_2(x3))
        x5 = F.relu(self.lin_3(x4))
        x6 = self.lin_4(x5)

        return x6




# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])
# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

net = CNN()

criterion = nn.CrossEntropyLoss()
l1_loss = nn.L1Loss(size_average=False)
optimizer = optim.Adam(net.parameters())

# node json data
nodes = []
for i in range(20):
    nodes.append({'id': i,
                  'layer': 1,
                  'graph': 0})
for i in range(20, 35):
    nodes.append({'id': i,
                  'layer': 2,
                  'graph': 0})
for i in range(35, 45):
    nodes.append({'id': i,
                  'layer': 3,
                  'graph': 0})
for i in range(45, 55):
    nodes.append({'id': i,
                  'layer': 4,
                  'graph': 0})

for i in range(55, 75):
    nodes.append({'id': i,
                  'layer': 1,
                  'graph': 1})
for i in range(75, 90):
    nodes.append({'id': i,
                  'layer': 2,
                  'graph': 1})
for i in range(90, 100):
    nodes.append({'id': i,
                  'layer': 3,
                  'graph': 1})
for i in range(45 + 55, 55 + 55):
    nodes.append({'id': i,
                  'layer': 4,
                  'graph': 1})

links = []
net_2 = CNN()
with torch.no_grad():
    p1, _ = net_2.lin_2.parameters()

    p1 *= 1000
    p1 = torch.abs(p1)
    for target, item in enumerate(p1):
        for source, num in enumerate(item):
            links.append({'source': source + 55,
                          'target': target + 20 + 55,
                          'value': num.item()
                          })

    p2, _ = net_2.lin_3.parameters()

    p2 *= 1000
    p2 = torch.abs(p2)
    for target, item in enumerate(p2):
        for source, num in enumerate(item):
            links.append({'source': source + 20 + 55,
                          'target': target + 35 + 55,
                          'value': num.item()
                          })

    p3, _ = net_2.lin_4.parameters()

    p3 *= 1000
    p3 = torch.abs(p3)
    for target, item in enumerate(p3):
        for source, num in enumerate(item):
            links.append({'source': source + 35 + 55,
                          'target': target + 45 + 55,
                          'value': num.item()
                          })

def validation(model, testloader, criterion):
    """
    Compute the accuracy and test loss.

    Accuracy computed using a running mean because the total number of training examples
    is not easily accessible the testloader object.
    """
    test_loss = 0
    accuracy = 0

    for images, labels in testloader:

        output = model(images)
        test_loss += criterion(output, labels).item()



        probs = F.softmax(output, dim=1)
        checker = probs.max(dim=1)
        correct = (labels.data == probs.max(dim=1)[1])
        accuracy += correct.type(torch.FloatTensor).mean()

    accuracy = accuracy / len(testloader)
    test_loss = test_loss / len(testloader)

    return test_loss, accuracy

epochs = 2
print_every = 100
steps = 0

# keeps track of json data
iteration_ID = 0

for e in range(epochs):

    net.train()
    running_loss = 0
    train_accuracy = 0
    for images, labels in trainloader:
        steps += 1
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)

        active_4 = F.softmax(output, dim=1)

        # l1 loss incorporation
        l1_norm = 0
        for param in net.parameters():
            l1_norm = l1_loss(param, target=torch.zeros_like(param))

        loss += l1_norm * 0.0005

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        probs = F.softmax(output, dim=1)
        """tensor.max() returns a 2 dimensional tensor.
        the first dimension contains the actual max values.
        The second returns the index of that max value."""
        correct = (labels.data == probs.max(dim=1)[1])
        train_accuracy += correct.type(torch.FloatTensor).mean()

        if steps % print_every == 0:
            net.eval()

            with torch.no_grad():


                test_loss, test_accuracy = validation(net, testloader, criterion)
            print("Epoch: {}/{}\n".format(e+1, epochs),
                  "Training Loss:      {:.3f}...".format(running_loss/print_every),
                  "Test Loss:      {:.3f}\n".format(test_loss),
                  "Training Accuracy:  {:.3f}...".format(train_accuracy/print_every),
                  "Test Accuracy:  {:.3f}".format(test_accuracy)
                  )
            running_loss = 0
            train_accuracy = 0
with torch.no_grad():

    # links = []
    p1, _ = net.lin_2.parameters()
    # p1 -= p1.min()
    # p1 /= p1.max()
    p1 = torch.abs(p1)
    p1 *= 100
    for target, item in enumerate(p1):
        for source, num in enumerate(item):
            links.append({'source': source,
                          'target': target + 20,
                          'value': num.item()
                          })

    p2, _ = net.lin_3.parameters()
    # p2 -= p2.min()
    # p2 /= p2.max()
    p2 = torch.abs(p2)
    p2 *= 100
    for target, item in enumerate(p2):
        for source, num in enumerate(item):
            links.append({'source': source + 20,
                          'target': target + 35,
                          'value': num.item()
                          })

    p3, _ = net.lin_4.parameters()
    # p3 -= p3.min()
    # p3 /= p3.max()
    p3 = torch.abs(p3)
    p3 *= 100
    for target, item in enumerate(p3):
        for source, num in enumerate(item):
            links.append({'source': source + 35,
                          'target': target + 45,
                          'value': num.item()
                          })

# save json data
with open('NN_data.json', 'w') as outfile:
    json.dump({"nodes": nodes,
               "links": links
               }, outfile)

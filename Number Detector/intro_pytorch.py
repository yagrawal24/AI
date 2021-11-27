import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training=True):
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST('./data', train=True, download=True,
                               transform=custom_transform)
    test_set = datasets.MNIST('./data', train=False,
                              transform=custom_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=50)

    if training:
        return train_loader
    else:
        return test_loader


def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    return model


def train_model(model, train_loader, criterion, T):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        running_loss = 0.0
        predict = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            total += train_loader.batch_size
            opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            _, index = torch.max(outputs, dim=1)
            predict += torch.sum(index == labels).item()
            running_loss += loss.item()

        print("Train Epoch: {}   Accuracy: {}/{}({:.2f}%) Loss: {:.3f}".format(epoch, predict, total, (predict * 100 /
                                                                                                       total),
                                                                               running_loss * train_loader.batch_size / total))


def evaluate_model(model, test_loader, criterion, show_loss=True):
    model.eval()

    predict = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            total += test_loader.batch_size
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, index = torch.max(outputs, dim=1)
            predict += torch.sum(index == labels).item()
            running_loss += loss.item()
    if show_loss:
        print("Average loss: " + "{:.4f}".format(running_loss * test_loader.batch_size / total))
    print("Accuracy: {:.2f}%".format((predict * 100) / total))


def predict_label(model, test_images, index):
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    image = test_images[index]
    predict_set = model(image)
    prob = F.softmax(predict_set, dim=1)
    prob = prob.tolist()
    for i in range(3):
        max = 0
        index = 0
        for j in range(len(prob[0])):
            if prob[0][j] > max:
                max = prob[0][j]
                index = j

        print("{}: {:.2f}%".format(class_names[index], max * 100))
        del prob[0][index]


def predict(model, image):
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    image = image
    predict_set = model(image)
    prob = F.softmax(predict_set, dim=1)
    prob = prob.tolist()
    for i in range(3):
        max = 0
        index = 0
        for j in range(len(prob[0])):
            if prob[0][j] > max:
                max = prob[0][j]
                index = j

        print("{}: {:.2f}%".format(class_names[index], max * 100))
        del prob[0][index]


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()

def main():
    train_loader = get_data_loader()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    predset, _ = iter(get_data_loader()).next()
    predict_label(model, predset, 1)
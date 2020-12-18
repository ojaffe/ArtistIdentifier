import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd

import os
import copy
import math
import numpy as np

data_dir = './data'

# Shows distribution of no. paintings
noPaintings = pd.read_csv(os.path.join(data_dir, 'artists.csv'), sep=',', index_col=0)
noPaintings = noPaintings.sort_values(by=['paintings'], ascending=True)
noPaintings.plot(kind='bar')
plt.ylabel('Paintings')
plt.xlabel('Artist ID')
plt.show()

# Data augmentation and norm. for training, just norm. for val+test
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

# Load data
full_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'images'))
class_names = full_dataset.classes

train_size = int(0.8 * len(full_dataset))
rem_size = math.floor((len(full_dataset) - train_size) / 2)
val_size = rem_size
test_size = rem_size

train_set, val_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

# Transform data
train_set.dataset.transform = data_transforms['train']
val_set.dataset.transform = data_transforms['val']
test_set.dataset.transform = data_transforms['test']

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Calculate class weights due to imbalanced dataset
noPaintings = pd.read_csv(os.path.join(data_dir, 'artists.csv'), sep=',')
largest_class = noPaintings['paintings'].max()
noClassData = noPaintings['paintings'].to_numpy()
noClassData = largest_class / noClassData


# Show some random training images
def show_image(img, title=None):
    img = img / 2 + 0.5  # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()


images, labels = next(iter(train_loader))
show_image(torchvision.utils.make_grid(images), title=[class_names[x] for x in labels])


# Sets up model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 14 * 14, 228)
        self.fc2 = nn.Linear(228, 50)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model_conv = Net()

# Load model
mode = 'load'  # Change to 'train'/'load'
saveDir = './models'

if mode == 'load':
    model_conv.load_state_dict(torch.load(os.path.join(saveDir, "standardCNN.pth")))

model_conv = model_conv.to(device)

noClassData = torch.Tensor(noClassData).to(device)
criterion = nn.CrossEntropyLoss(weight=noClassData)
optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.001, betas=(0.9, 0.999))


def train_model(model, criterion, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Arrays for plotting data
    train_loss_arr = []
    train_acc_arr = []
    val_acc_arr = []

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Training
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward, backward, optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / train_size
        epoch_train_acc = running_corrects.double() / train_size
        train_loss_arr.append(epoch_train_loss)
        train_acc_arr.append((epoch_train_acc))

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'Train', epoch_train_loss, epoch_train_acc))

        # Validation
        running_corrects = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # Statistics
                running_corrects += torch.sum(preds == labels.data)

        epoch_val_acc = running_corrects.double() / val_size
        val_acc_arr.append(epoch_val_acc)

        print('{} Acc: {:.4f}'.format(
            'Val', epoch_val_acc))

        # deep copy the model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # Plot data
    plt.plot(train_acc_arr)
    plt.plot(val_acc_arr)
    plt.title('Accuracy over epochs')
    plt.show()

    plt.plot(train_loss_arr)
    plt.title('Training loss over epochs')
    plt.show()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model):
    running_corrects = 0

    confusion_matrix = np.zeros((len(class_names), len(class_names)))
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Statistics
            running_corrects += torch.sum(preds == labels.data)

            # Update confusion matrix
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            for x in range(len(preds)):
                confusion_matrix[labels[x]][preds[x]] += 1

    epoch_acc = running_corrects.double() / test_size
    print('{} Test Acc: {:.4f}'.format('Test', epoch_acc))

    plot_confusion_matrix(confusion_matrix)


def plot_confusion_matrix(confusion_matrix):
    # Absolute predictions
    fig, ax = plt.subplots(1)

    ax.set_title('No. Predictions Confusion Matrix')
    ax.matshow(confusion_matrix)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    plt.show()

    # Percentage predictions
    for i in range(len(class_names)):
        totalPredicted = sum(confusion_matrix[:, i])
        if totalPredicted == 0:
            continue

        confusion_matrix[:, i] = confusion_matrix[:, i] / totalPredicted

    fig, ax = plt.subplots(1)

    ax.set_title('Percentage Predictions Confusion Matrix')
    ax.matshow(confusion_matrix)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    plt.show()


pytorch_total_params = sum(p.numel() for p in model_conv.parameters())
print(pytorch_total_params)

# Train and save model
if mode == 'train':
    model_conv = train_model(model_conv, criterion, optimizer_conv, num_epochs=10)
    torch.save(model_conv.state_dict(), os.path.join(saveDir, "standardCNN.pth"))

test_model(model_conv)

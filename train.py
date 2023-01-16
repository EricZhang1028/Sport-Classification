from utils.loader import SportLoader
from utils.network import Network
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import time
# import logging

# logging.basicConfig(level=logging.DEBUG, filename=f"log_{int(time.time())}.txt")

CPT_FILE = "HW1_310551135.pt"

num_classes = 10
num_epochs = 100
batch_size = 64
learning_rate = 0.008

# logging.info(f"num_epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_dataset = SportLoader("train", transform=transform)
valid_dataset = SportLoader("val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

model = Network(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)

histroy = {}
histroy["train_loss"] = []
histroy["valid_loss"] = []
histroy["train_acc"] = []
histroy["valid_acc"] = []

total_step = len(train_loader)
min_valid_loss = float("inf")

for epoch in range(num_epochs):
    train_loss, valid_loss = 0, 0
    train_total, train_correct = 0, 0

    model.train()
    for i,(images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # compute acc
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss
        train_loss += loss.item()

    avg_train_loss = train_loss/len(train_loader)
    histroy["train_loss"].append(round(avg_train_loss, 2))
    histroy["train_acc"].append(round(100 * train_correct / train_total, 2))
    # logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    print ('[Train] Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    model.eval()
    with torch.no_grad():
        valid_correct = 0
        valid_total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()
            valid_loss += loss.item()

            del images, labels, outputs
        
        # logging.info('Accuracy of the network on the validation images: {:.4f} %'.format(100 * valid_correct / valid_total))
        print('[Valid] Accuracy of the network on the validation images: {:.4f} %'.format(100 * valid_correct / valid_total))

    avg_valid_loss = valid_loss / len(valid_loader)
    if avg_valid_loss < min_valid_loss:
        torch.save(model.state_dict(), CPT_FILE)
        min_valid_loss = avg_valid_loss
        print(f"=============Model Saved in Epoch {epoch+1}")

    histroy["valid_loss"].append(round(avg_valid_loss, 2))
    histroy["valid_acc"].append(round(100 * valid_correct / valid_total, 2))

# logging.info(f"train_loss: {histroy['train_loss']}")
# logging.info(f"valid_loss: {histroy['valid_loss']}")
# logging.info(f"train_acc: {histroy['train_acc']}")
# logging.info(f"valid_acc: {histroy['valid_acc']}")

# epoch_count = range(1, len(histroy["train_loss"])+1)
# plt.plot(epoch_count, histroy["train_loss"])
# plt.plot(epoch_count, histroy["valid_loss"])
# plt.legend(['train', 'valid'])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

# epoch_count = range(1, len(histroy["train_acc"])+1)
# plt.plot(epoch_count, histroy["train_acc"])
# plt.plot(epoch_count, histroy["valid_acc"])
# plt.legend(['train_acc', 'valid_acc'])
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()
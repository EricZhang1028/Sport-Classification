from utils.loader import SportLoader
from utils.network import Network
from torch.utils.data import DataLoader
from torchvision import transforms

import pandas as pd
import torch

TEST_FILE = "HW1_310551135.csv"

num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_dataset = SportLoader("test", transform=transform)
test_loader = DataLoader(test_dataset)

model = Network(num_classes).to(device)
model.load_state_dict(torch.load("HW1_310551135.pt"))

img_list, pred_list = [], []
model.eval()
with torch.no_grad():
    # for i, (images, _) in enumerate(test_loader):
    #     images = images.to(device)
    #     output = model(images.permute(0, 3, 1, 2).float())
    #     _, predicted = torch.max(output.data, 1)

    #     img_list.append(test_loader.dataset.img_name[i])
    #     pred_list.append(predicted.item())
    
    ''' out here '''
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        img_list.append(test_loader.dataset.img_name[i])
        pred_list.append(predicted.item())
    print('Accuracy of the network on the testing images: {:.4f} %'.format(100 * correct / total))
    ''' out here '''

df = pd.DataFrame(list(zip(img_list, pred_list)))
df.to_csv(TEST_FILE, index=False, header=["names", "label"])

number_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"number_of_paramss = {number_of_params}")
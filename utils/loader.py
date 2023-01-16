from torch.utils import data
from skimage import io

import pandas as pd
import os

class SportLoader(data.Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode
        if self.mode in ["train", "val"]:
            self.sport = pd.read_csv(self.mode+".csv")
            self.img_name = self.sport["names"].values.tolist()
            self.label = self.sport["label"].values.tolist()
        else:
            self.img_name = sorted(os.listdir("test"), key=lambda x: int(x.split(".")[0]))
            self.label = ["-1" for _ in range(len(self.img_name))]

        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        image_path = os.path.join(self.mode, self.img_name[index])
        self.img = io.imread(image_path)
        self.target = self.label[index]

        if self.transform:
            self.img = self.transform(self.img)
        
        return self.img, self.target
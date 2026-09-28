import os 
import numpy as np
from PIL import Image
from torch.utils.data import Dataset 
from torchvision import transforms 

root_dir = "ocr/chars"

class CharacterDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name:idx for idx, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_folder = os.path.join(root_dir,cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for filename in os.listdir(cls_folder):
                filepath = os.path.join(cls_folder,filename)
                self.samples.append((filepath,self.class_to_idx[cls_name]))
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        filepath,label= self.samples[idx]
        image = Image.open(filepath).convert("L")
        if self.transform:
            image = self.transform(image)
        return image,label    

transform = transforms.Compose([transforms.Resize((32,32)),
                                transforms.ToTensor()])
dataset=CharacterDataset(root_dir,transform=transform)
print(len(dataset))
image,label = dataset[4]
print(image.shape,label)

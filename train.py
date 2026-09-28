import torch
import torch.nn as nn 
from torch.utils.data import DataLoader 
from dataset import CharacterDataset 
from model import charCNN
from dataset import transform 

root_dir = "ocr/chars"
dataset = CharacterDataset(root_dir,transform=transform)
dataloader = DataLoader(dataset,batch_size=32,shuffle=True)

model = charCNN(num_classes=len(dataset.class_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

num_epochs = 10 

for epoch in range(num_epochs):
    running_loss = 0.00
    correct = 0 
    total = 0

    for images,labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

epoch_loss = running_loss / len(dataloader)
epoch_acc = 100 * correct / total
print(f"Epoch{epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset
import pandas as pd
import numpy as np
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


data_dir = './data/Petimages'

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        

        self.dropout = nn.Dropout(0.35)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  
        self.fc2 = nn.Linear(512, 256)  
        self.fc3 = nn.Linear(256,2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 16 * 16)  
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = ConvNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)  


train_losses = []


# Training Loop
num_epochs = 5  

for epoch in range(num_epochs):
    print(f'--- Epoch {epoch + 1}/{num_epochs} ---')
    model.train()  
    running_loss = 0.0
    
    # Iterate over the training data in batches
    for batch_idx, (inputs, labels) in enumerate(train_loader):
      
        inputs, labels = inputs.to(device), labels.to(device)
        
        
        optimizer.zero_grad()
        
       
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
       
        running_loss += loss.item()

        # Print progress every 10 batches (or adjust as needed)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    scheduler.step()

    # Calculate average loss for this epoch
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}")
    
    


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
print(f"Validation Accuracy: {100 * correct / total:.2f}%")


#torch.save(model.state_dict(), 'model_state_dict.pth')
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Progression During Training')
plt.legend()
plt.grid(True)
plt.show()
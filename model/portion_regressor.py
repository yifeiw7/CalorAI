import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import os
import json
import numpy as np
from PIL import Image

#  paths current directory is calorai 
DATA_PATH = "data"
MODEL_PATH = "model"
TRAIN_FILE = os.path.join(DATA_PATH, "train.json")
CALORIE_DB_FILE = os.path.join(DATA_PATH, "calories_database.json")

os.makedirs(MODEL_PATH, exist_ok=True)

# Load database
with open(CALORIE_DB_FILE, "r") as f:
    calorie_db = json.load(f)

FOOD_LABELS = sorted(list(calorie_db.keys()))
NUM_CLASSES = len(FOOD_LABELS)

# dataset
class FoodPortionDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["name"] + ".png")
        image = Image.open(img_path).convert("RGB")

        # Convert food types to a one-hot encoded vector
        food_vector = torch.zeros(NUM_CLASSES)
        portion_vector = torch.zeros(NUM_CLASSES)

        for food, portion in zip(item["food type"], item["portion"]):
            if food in FOOD_LABELS:
                food_idx = FOOD_LABELS.index(food)
                food_vector[food_idx] = 1  
                portion_vector[food_idx] = float(portion)  

        if self.transform:
            image = self.transform(image)

        return image, food_vector, portion_vector 

transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class PortionRegressor(nn.Module):
    def __init__(self, num_classes):
        super(PortionRegressor, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  
        
        self.fc = nn.Sequential(
            nn.Linear(512 + num_classes, 128),  
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  
        )

    def forward(self, img, food_vector):
        features = self.backbone(img) 
        x = torch.cat((features, food_vector), dim=1) 
        return self.fc(x) 

if __name__ == "__main__":
    print("Training portion regressor...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PortionRegressor(NUM_CLASSES).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = FoodPortionDataset(json_path=os.path.join(DATA_PATH, "train.json"), 
                                       img_dir=os.path.join(DATA_PATH, "images"),
                                       transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    EPOCHS = 100

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, food_vectors, portions in train_loader:
            images, food_vectors, portions = images.to(device), food_vectors.to(device), portions.to(device)

            optimizer.zero_grad()
            outputs = model(images, food_vectors) 
            loss = criterion(outputs, portions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": NUM_CLASSES  
    }

    torch.save(checkpoint, os.path.join(MODEL_PATH, "portion_regressor.pth"))
    print(f"Portion regressor model saved successfully with NUM_CLASSES = {NUM_CLASSES}!")

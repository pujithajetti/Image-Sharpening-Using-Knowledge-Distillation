# train.py
import os, glob
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from models import StudentNet
import lpips

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

class SharpenDataset(Dataset):
    def __init__(self, sharp_dir, blurry_dir):
        self.sharp_paths = sorted(glob.glob(f"{sharp_dir}/*"))
        self.blurry_paths = sorted(glob.glob(f"{blurry_dir}/*"))

    def __len__(self):
        return len(self.sharp_paths)

    def __getitem__(self, idx):
        sharp = transform(Image.open(self.sharp_paths[idx]).convert('RGB'))
        blurry = transform(Image.open(self.blurry_paths[idx]).convert('RGB'))
        return blurry, sharp

# Load dataset
dataset = SharpenDataset("datasets/original", "datasets/noisy")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Init model
model = StudentNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss()
perceptual = lpips.LPIPS(net='alex').cuda()

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for blurry, sharp in dataloader:
        blurry, sharp = blurry.cuda(), sharp.cuda()
        pred = model(blurry)
        loss = loss_fn(pred, sharp) + perceptual(pred, sharp).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss/len(dataloader):.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/student_model.pth")

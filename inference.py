# inference.py
import torch
from models import StudentNet
import numpy as np
from PIL import Image
import torchvision.transforms as T

model = StudentNet()
model.load_state_dict(torch.load("models/student_model.pth", map_location="cpu"))
model.eval()

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

inv_transform = T.ToPILImage()

def sharpen_image(img):
    orig_size = img.size
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x).squeeze(0)
    out = inv_transform(out.clamp(0, 1))
    return out.resize(orig_size)

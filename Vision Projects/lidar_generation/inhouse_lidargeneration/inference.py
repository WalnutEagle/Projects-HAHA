import torch
from model import DepthEstimationModel
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

def load_model(model_path):
    model = DepthEstimationModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def infer_depth(image_path, model):
    image = Image.open(image_path).convert("RGB")
    transform = ToTensor()
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        depth = model(image)
    return depth.squeeze().numpy()

if __name__ == "__main__":
    model = load_model("depth_model.pth")
    depth_map = infer_depth("path_to_image.jpg", model)
    print(depth_map)

import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class DepthEstimator:
    def __init__(self, model_type="DPT_Large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def estimate_depth(self, frame):
        input_batch = self.transform(frame).to(self.device)
        with torch.no_grad():
            depth = self.model(input_batch)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        return depth

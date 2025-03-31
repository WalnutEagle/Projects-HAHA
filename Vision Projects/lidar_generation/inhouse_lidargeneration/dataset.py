import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(os.listdir(os.path.join(root_dir, "image")))
        self.depth_paths = sorted(os.listdir(os.path.join(root_dir, "depth")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "image", self.image_paths[idx])
        depth_path = os.path.join(self.root_dir, "depth", self.depth_paths[idx])

        image = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path)

        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)

        return image, np.array(depth, dtype=np.float32)

def get_kitti_dataloader(root_dir, batch_size=8, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = KITTIDataset(root_dir, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

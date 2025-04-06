import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2

class COCODataset(Dataset):
    def __init__(self, dataset_dir, subset):
        self.coco = COCO(os.path.join(dataset_dir, f"annotations/instances_{subset}2017.json"))
        self.image_dir = os.path.join(dataset_dir, f"{subset}2017")
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        masks = torch.zeros((224, 224, len(anns)), dtype=torch.float32)
        labels = []
        for i, ann in enumerate(anns):
            mask = self.coco.annToMask(ann)
            mask = cv2.resize(mask, (224, 224))
            masks[:, :, i] = torch.tensor(mask, dtype=torch.float32)
            labels.append(ann['category_id'])

        labels = torch.tensor(labels, dtype=torch.long)
        return image, (labels, masks)

class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        cls_output = self.cls_head(features)
        mask_output = self.mask_head(features)
        return cls_output, mask_output

def main():
    dataset_dir = "path/to/coco"  # Replace with the path to your COCO dataset
    num_classes = 81  # COCO has 80 classes + background

    train_dataset = COCODataset(dataset_dir, "train")
    val_dataset = COCODataset(dataset_dir, "val")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = MaskRCNN(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_mask = nn.BCEWithLogitsLoss()

    for epoch in range(50):
        model.train()
        for images, (labels, masks) in train_loader:
            optimizer.zero_grad()
            cls_output, mask_output = model(images)
            loss_cls = criterion_cls(cls_output, labels)
            loss_mask = criterion_mask(mask_output, masks)
            loss = loss_cls + loss_mask
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "inhouse_mask_rcnn.pth")
    print("Training complete. Model saved as inhouse_mask_rcnn.pth.")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_kitti_dataloader
from model import DepthEstimationModel

# Novel Loss Function: Combination of Scale-Invariant Loss and Structural Similarity Index (SSIM)
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Scale-Invariant Loss
        diff = pred - target
        scale_invariant_loss = torch.mean(diff ** 2) - 0.85 * torch.mean(diff) ** 2

        # SSIM Loss
        ssim_loss = 1 - self.ssim(pred, target)

        return scale_invariant_loss + ssim_loss

    def ssim(self, pred, target):
        # Simplified SSIM implementation
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_pred = torch.mean(pred)
        mu_target = torch.mean(target)
        sigma_pred = torch.var(pred)
        sigma_target = torch.var(target)
        sigma_pred_target = torch.mean((pred - mu_pred) * (target - mu_target))
        ssim = (2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2) / (
            (mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2)
        )
        return ssim

def train_model(root_dir, epochs=20, batch_size=8, lr=1e-4, val_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthEstimationModel().to(device)
    dataloader = get_kitti_dataloader(root_dir, batch_size=batch_size)
    
    # Split dataset into training and validation
    dataset_size = len(dataloader.dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataloader.dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = DepthLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        # Training Loop
        model.train()
        train_loss = 0
        for images, depths in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, depths.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, depths in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                images, depths = images.to(device), depths.to(device)
                outputs = model(images)
                loss = criterion(outputs, depths.unsqueeze(1))
                val_loss += loss.item()

        scheduler.step()  # Adjust learning rate
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")

    torch.save(model.state_dict(), "depth_model.pth")
    print("Model saved as depth_model.pth")

if __name__ == "__main__":
    train_model(root_dir="path_to_kitti_dataset")

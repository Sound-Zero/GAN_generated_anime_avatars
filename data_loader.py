import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from config import Config

class AnimeFaceDataset(Dataset):
    def __init__(self, img_dir, img_size=128, transform=None):
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        self.image_paths = []
        self.count =0
        # 扫描所有jpg文件
        for file in os.listdir(img_dir):
            if file.lower().endswith('.jpg'):
                self.count +=1
                self.image_paths.append(os.path.join(img_dir, file))
            # if self.count == 320:
            #     break
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def get_dataloader(batch_size=32, num_workers=4, img_size=128):
    config = Config()
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 到 [-1, 1]
    ])
    
    dataset = AnimeFaceDataset(config.img_dir, img_size=img_size, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

# 测试 (可选)
if __name__ == "__main__":
    config = Config()
    dataloader = get_dataloader(batch_size=config.batch_size, img_size=128)
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")  # 应为 [32, 3, 128, 128]
        break
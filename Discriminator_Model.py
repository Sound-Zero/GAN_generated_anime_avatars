import torch
import torch.nn as nn
from config import Config

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=256):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 卷积块
        self.block1 = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 1, 8, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 权重初始化
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.view(-1, 1)  # 输出 (batch_size, 1)

# 测试实例 (可选)
if __name__ == "__main__":
    config = Config()
    disc = Discriminator(img_channels=3, img_size=256)
    real = torch.randn(32, 3, 256, 256)
    pred = disc(real)
    print(f"Discriminator output shape: {pred.shape}")  # 应为 [32, 1]
import torch
import torch.nn as nn
from config import Config

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, img_size=256):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 初始投影层: z -> 512 x 16 x 16
        self.fc = nn.Linear(z_dim, 512 * 16 * 16)
        
        # 转置卷积块
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
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
        #import sys
        #print(f"In Generator forward: input x.shape = {x.shape}")
        #sys.stdout.flush()
        x = self.fc(x)
        x = x.view(-1, 512, 16, 16)     #(batch_size, 512, 16, 16)
        x = self.block1(x)              #(batch_size, 256, 32, 32)
        x = self.block2(x)              #(batch_size, 128, 64, 64)
        x = self.block3(x)              #(batch_size, 64, 128, 128)
        x = self.block4(x)              #(batch_size, 32, 256, 256)

        # print(f"Generator output shape: {x.shape}")
        # sys.stdout.flush()
        return x

# 测试实例 (可选)
if __name__ == "__main__":
    config = Config()
    gen = Generator(z_dim=100)
    z = torch.randn(32, 100)
    fake = gen(z)
    print(f"Generated shape: {fake.shape}")  # 应为 [32, 3, 256,256]
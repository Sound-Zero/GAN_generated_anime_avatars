import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torchvision.utils import save_image
from Generator_Model import Generator
from Discriminator_Model import Discriminator
from data_loader import get_dataloader
from config import Config
import argparse
import logging
import time
from tqdm import tqdm
import sys


def train_gan(num_epochs=3000, z_dim=100, device=None, log_interval=100):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = Config
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.generater_model_dir, exist_ok=True)
    os.makedirs(config.discriminator_model_dir, exist_ok=True)
    
    # 初始化 logger
    logger = logging.getLogger('GAN_Train')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(config.log_dir, 'train.log'),encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter.encoding = 'utf-8'   # 3.9+ 支持
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    last_log_time = time.time()
    
    # 初始化模型
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator(img_channels=3, img_size=256).to(device)
    
    # WGAN-GP 参数
    lambda_gp = 10
    
    # 优化器 (WGAN-GP 推荐RMSprop，但继续用Adam)
    g_optimizer = optim.Adam(G.parameters(), lr=config.start_lr, betas=(0.0, 0.9))
    d_optimizer = optim.Adam(D.parameters(), lr=config.start_lr, betas=(0.0, 0.9))
    
    # 学习率调度器
    g_scheduler = LinearLR(g_optimizer, start_factor=1.0, end_factor=config.target_lr / config.start_lr, total_iters=config.lr_schedule)
    d_scheduler = LinearLR(d_optimizer, start_factor=1.0, end_factor=config.target_lr / config.start_lr, total_iters=config.lr_schedule)
    
    # 数据加载器 (使用256*256 匹配模型)
    dataloader = get_dataloader(batch_size=config.batch_size, img_size=256)
    
    # 固定噪声用于监控
    fixed_z = torch.randn(64, z_dim, device=device)
    print(f"Fixed z shape: {fixed_z.shape}")
    
    print(f"开始训练GAN，使用设备: {device}")
    print(f"数据集大小: {len(dataloader.dataset)}")
    
    for epoch in range(num_epochs):

        print(f"Epoch [{epoch}/{num_epochs}]")
        for batch_idx, real_imgs in tqdm(enumerate(dataloader)):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)    # (batch_size, 3, 256, 256)
            
            # D训练 (n_critic=2次，WGAN-GP)
            for _ in range(2):
                # D on real and fake (WGAN)
                D.zero_grad()
                d_real = D(real_imgs)
                z = torch.randn(batch_size, z_dim, device=device)
                fake_imgs = G(z).detach()
                d_fake = D(fake_imgs)
                
                # WGAN D loss
                d_loss = -torch.mean(d_real) + torch.mean(d_fake)
                
                # Gradient Penalty
                gp = gradient_penalty(D, real_imgs, fake_imgs, device, batch_size)
                d_loss += lambda_gp * gp
                
                d_loss.backward()
                d_optimizer.step()
            
            # G训练 (1次，WGAN)
            G.zero_grad()
            z = torch.randn(batch_size, z_dim, device=device)
            fake_imgs = G(z)
            d_fake = D(fake_imgs)
            g_loss = -torch.mean(d_fake)  # WGAN G loss
            g_loss.backward()
            g_optimizer.step()
            
            # 调度器步进 (每batch)
            g_scheduler.step()
            d_scheduler.step()
            
            # 时间-based 日志检查
            current_time = time.time()
            if current_time - last_log_time >= config.log_save_time:
                logger.info(f"当前搜集到的指标 - Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                            f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
                # 刷新日志文件
                for handler in logger.handlers:
                    handler.flush()
                last_log_time = current_time
            
            # 日志
            if batch_idx % log_interval == 0:
                logger.info(f'Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] '
                            f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')
        
        # Epoch末: 生成固定样本并保存
        if epoch % 5 == 0:
            with torch.no_grad():
                fake_fixed = G(fixed_z)
                # 反归一化保存
                fake_fixed = (fake_fixed + 1) / 2
                save_image(fake_fixed, os.path.join(config.log_dir, f'epoch_{epoch}.png'), nrow=8)
        
        # 保存模型 (每10 epochs)
        if epoch % 10 == 0:
            torch.save(G.state_dict(), os.path.join(config.generater_model_dir, f'G_epoch_{epoch}.pth'))
            torch.save(D.state_dict(), os.path.join(config.discriminator_model_dir, f'D_epoch_{epoch}.pth'))
            print(f'模型已保存于 epoch {epoch}')
    
    print("训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练GAN生成动漫头像')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数 (默认1用于测试)')
    parser.add_argument('--z_dim', type=int, default=100, help='噪声维度')
    parser.add_argument('--log_interval', type=int, default=100, help='日志间隔')
    args = parser.parse_args()
    
    def gradient_penalty(D, real_imgs, fake_imgs, device, batch_size):
        """计算梯度惩罚"""
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolates = (alpha * real_imgs + ((1 - alpha) * fake_imgs)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones([batch_size, 1], device=device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1) ** 2)
        return penalty

    train_gan(num_epochs=args.num_epochs, z_dim=args.z_dim, log_interval=args.log_interval)
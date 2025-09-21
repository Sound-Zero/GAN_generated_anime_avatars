import os
import argparse
import torch
from torchvision.utils import save_image
from Generator_Model import Generator
from config import Config

def generate_images(model_path=None, num_samples=64, z_dim=100, output_path=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = Config
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 初始化并加载模型
    G = Generator(z_dim=z_dim).to(device)
    if model_path:
        G.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # 默认加载最新模型 (假设训练后有文件)
        latest_model = max([f for f in os.listdir(config.generater_model_dir) if f.endswith('.pth')], 
                           key=lambda x: int(x.split('_')[-1].split('.')[0]), default=None)
        if latest_model:
            model_path = os.path.join(config.generater_model_dir, latest_model)
            G.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise ValueError("未找到训练模型，请指定--model_path")
    
    G.eval()
    
    # 生成噪声
    z = torch.randn(num_samples, z_dim, 1, 1, device=device)
    
    # 生成图片
    with torch.no_grad():
        fake_imgs = G(z)
        # 反归一化到 [0,1]
        fake_imgs = (fake_imgs + 1) / 2
    
    # 保存
    if output_path is None:
        output_path = os.path.join(config.log_dir, f'generated_anime_{num_samples}.png')
    save_image(fake_imgs, output_path, nrow=8, normalize=True)
    
    print(f"生成 {num_samples} 张动漫头像，保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAN生成动漫头像')
    parser.add_argument('--num_samples', type=int, default=64, help='生成样本数')
    parser.add_argument('--model_path', type=str, default=None, help='Generator模型路径')
    parser.add_argument('--output_path', type=str, default=None, help='输出图片路径')
    args = parser.parse_args()
    
    generate_images(args.model_path, args.num_samples, output_path=args.output_path)
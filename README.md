# GAN 生成动漫头像项目

## 项目概述
这是一个基于 PyTorch 的 GAN (Generative Adversarial Network) 项目，用于生成动漫风格的人脸头像。最初使用 DCGAN 架构训练，但由于训练效果变差（生成图像从 epoch 10 后开始模糊），进行了优化改进，包括超参数调整和切换到 WGAN-GP 损失函数，以提高训练稳定性和生成质量。

项目目标：使用动漫脸数据集训练生成器 (Generator) 和判别器 (Discriminator)，生成高质量的动漫头像。

## 环境要求
- Python 3.8+
- PyTorch 1.10+ (支持 CUDA，如果使用 GPU)
- torchvision
- Pillow (PIL)
- tqdm
- argparse

安装依赖：
```
pip install torch torchvision pillow tqdm
```

## 项目结构
- `config.py`: 配置参数，包括学习率、batch_size、路径等。
- `data_loader.py`: 数据加载器，支持加载图像数据集并预处理 (Resize 到 256x256, Normalize 到 [-1, 1])。
- `Generator_Model.py`: 生成器模型，使用转置卷积从噪声向量生成 256x256x3 图像。
- `Discriminator_Model.py`: 判别器模型，使用卷积判断图像真伪 (WGAN-GP 版本，无 Sigmoid 输出)。
- `train.py`: 训练脚本，支持 WGAN-GP 损失、梯度惩罚、学习率调度。
- `generate.py`: 生成图像脚本 (使用训练好的模型生成新图像)。
- `data/faces_1w/`: 动漫脸数据集 (约 1w 张图像，路径在 config.py 中定义)。
- `log/`: 训练日志和生成图像 (e.g., epoch_*.png, train.log)。
- `model/generator/` 和 `model/discriminator/`: 保存的模型权重 (e.g., G_epoch_*.pth)。

## 训练步骤
1. 确保数据集在 `./data/faces_1w/` 目录下 (动漫脸图像，.jpg 格式)。
2. 运行训练：
   ```
   python train.py --num_epochs 3000 --z_dim 100 --log_interval 100
   ```
   - `--num_epochs`: 训练轮数 (默认 3000)。
   - `--z_dim`: 噪声维度 (默认 100)。
   - `--log_interval`: 日志输出间隔 (默认 100 batches)。

训练过程：
- 每 epoch 结束时 (每 5 epochs)，保存生成图像到 `log/epoch_*.png` 和模型到 `model/`。
- 日志记录 D_loss 和 G_loss 到 `log/train.log`。
- 使用 GPU (如果可用) 加速训练。

## 生成图像
使用训练好的模型生成新图像：
```
python generate.py --num_samples 64 --output_dir ./generated_images/
```
- 生成 64 张图像保存到指定目录。

## 改进说明
### 原始问题
- 训练效果从 epoch 10 后变差：生成图像模糊/畸形。
- 损失曲线：D_loss 快速下降 (D 过强)，G_loss 先降后升，导致模式崩溃。
- 原因：学习率衰减过快 (total_iters=3000，但实际步数更多)；D:G 训练比例 5:1 过高；BCE 损失易梯度消失；数据集域适配 (虽为动漫脸，但需确认质量)。

### 优化措施
1. **超参数调整** (`config.py` 和 `train.py`)：
   - 学习率调度：`lr_schedule` 从 3000 改为 100000 (匹配预期总步数 ~1000 epochs * 199 batches)。
   - D:G 比例：从 5:1 降到 2:1，平衡训练。
   - Adam betas：调整为 (0.0, 0.9)，适合 WGAN。

2. **损失函数切换** (`train.py`)：
   - 从 BCE Loss 切换到 WGAN-GP (Wasserstein GAN with Gradient Penalty)。
   - D 输出移除 Sigmoid (`Discriminator_Model.py`)。
   - 添加梯度惩罚 (lambda_gp=10)，防止梯度爆炸，提高稳定性。
   - D_loss = -mean(D(real)) + mean(D(fake)) + GP。
   - G_loss = -mean(D(fake))。

3. **数据适配**：
   - 确认数据集为动漫脸 (`./data/faces_1w`)，无需额外替换。
   - 数据加载：随机翻转增强，Resize 到 256x256。

### 预期效果
- 训练更稳定，G_loss 和 D_loss 保持平衡。
- 生成图像从早期模糊问题改善，接近真实动漫风格。
- 建议监控 `log/train.log` 和 `log/epoch_*.png` 验证 (e.g., 到 epoch 50 应见改善)。

## 故障排除
- **训练慢**：使用 GPU，减小 batch_size (config.py)。
- **内存不足**：降低 batch_size 或 img_size。
- **生成质量差**：增加 epochs，检查数据集多样性，或微调 lr。
- **日志无输出**：检查 `log_save_time=60` (每 60s 保存)。

## 未来改进
- 目前训练过程D_loss 和 G_loss 波动极大，需要更先进的方法改进。

## 其他
- 数据集：使用公共动漫脸数据集https://aistudio.baidu.com/datasetdetail/110820。
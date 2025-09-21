class Config:
    img_dir = './data/faces_1w'  # 图片文件夹路径
    img_size = (256, 256)        # 图片原始大小 (匹配模型)
    img_generate_size = (256, 256)
    
    batch_size = 32
    
    z_dim = 100                  # 噪声维度
    num_epochs = 3000            # 训练轮数
    
    start_lr = 6e-4
    target_lr = 1e-5
    lr_schedule = 100000         # 学习率调度总步数，基于预期总训练步数调整
    
    log_dir = './log'            # 日志文件夹路径
    log_save_time = 60           # 每60秒保存一次日志
    generater_model_dir = './model/generator'  # 生成器模型保存路径 (修正拼写)
    discriminator_model_dir = './model/discriminator'  # 判别器模型保存路径